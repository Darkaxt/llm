"""Named OpenAI-compatible APISIX profiles and synthetic availability probes."""

import json
import os
import platform
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import click
import httpx2
import openai

import llm
from llm.default_plugins.openai_models import Chat

_PROBE_ERRORS = (openai.APIError, ValueError, TypeError, AttributeError)


def _config_path():
    return llm.user_dir() / "apisix.json"


def _config():
    path = _config_path()
    if not path.exists():
        return {"profiles": {}}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("profiles"), dict):
            raise TypeError
        return config
    except (ValueError, TypeError, OSError) as ex:
        raise click.ClickException("Cannot read APISIX profile configuration") from ex


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _profile(name):
    profile = _config()["profiles"].get(name)
    if not isinstance(profile, dict):
        raise click.ClickException("Unknown APISIX profile; create it first")
    return profile


def _key_env(name):
    return "LLM_APISIX_{}_API_KEY".format(re.sub("[^A-Z0-9]", "_", name.upper()))


def _client(name, profile, key=None):
    api_key = llm.get_key(input=key, alias=profile["key_alias"], env=_key_env(name))
    if not api_key:
        raise click.ClickException(
            "No profile key. Run: llm keys set {}".format(profile["key_alias"])
        )
    return openai.OpenAI(
        base_url=profile["base_url"],
        api_key=api_key,
        timeout=None,
        max_retries=0,
        http_client=httpx2.Client(
            verify=profile.get("verify_ssl", True), timeout=None, follow_redirects=False
        ),
    )


def _safe_error(ex):
    # Never include server bodies, request URLs, or exception messages in artifacts.
    result = {"status": "failed", "error_type": type(ex).__name__}
    if isinstance(ex, openai.APIStatusError):
        result["http_status"] = ex.status_code
    return result


class ApisixChat(Chat):
    def __init__(self, name, profile, model):
        super().__init__(
            f"apisix/{name}/{model}",
            model_name=model,
            reasoning=True,
            supports_tools=True,
        )
        self.profile_name = name
        self.profile = profile
        self.needs_key = profile["key_alias"]
        self.key_env_var = _key_env(name)

    def __str__(self):
        return f"APISIX Chat: {self.model_id}"

    def get_client(self, key, *, async_=False):
        return _client(self.profile_name, self.profile, key)

    def execute(self, prompt, stream, response, conversation=None, key=None):
        try:
            yield from super().execute(prompt, stream, response, conversation, key)
        except openai.APIError as ex:
            detail = _safe_error(ex)
            raise llm.ModelError(f"APISIX request failed: {detail}") from None


@llm.hookimpl
def register_models(register):
    for name, profile in _config()["profiles"].items():
        for model in profile.get("models", []):
            register(ApisixChat(name, profile, model))


def _discover(client):
    # A single request, without SDK auto-pagination or guessed model IDs.
    page = client.models.list()
    if not isinstance(page.data, list):
        raise TypeError("Invalid model discovery response")
    models = sorted(
        {item.id for item in page.data if isinstance(item.id, str) and item.id}
    )
    if not models:
        raise ValueError("Empty model discovery")
    return models


def _cache_models(name, models):
    config = _config()
    config["profiles"][name]["models"] = models
    _write_json(_config_path(), config)


def inspect_knowledge(directory):
    """Read the existing local virtual bundle; return structure, never contents."""
    if not directory:
        return {"status": "not_configured"}
    from llm.default_plugins.openwebui import (
        _local_knowledge_manifest,
        _tide_v3_macro_pointers,
    )

    try:
        manifest = _local_knowledge_manifest(Path(directory))
        entries = {
            "/".join(filter(None, (item["path"], item["filename"]))): item
            for item in manifest
        }

        def content(path):
            entry = entries[path]
            data = entry.get("_content")
            if data is None:
                data = Path(entry["_local_path"]).read_bytes()
            return data.decode("utf-8-sig")

        problems = []
        for instruction in ("SKILL.md", "SPEC.md"):
            candidates = [p for p in entries if p.split("/")[-1] == instruction]
            if not candidates or not any(content(p).strip() for p in candidates):
                problems.append("missing_instruction:" + instruction)
        rule_dirs = sorted(
            {
                p[: match.end() - 1]
                for p in entries
                if (match := re.search(r"(?:^|/)rules/[^/]+/", p))
            }
        )
        pending = []
        for rule in rule_dirs:
            prefix = rule.rsplit("rules/", 1)[0]
            for filename in ("detection.json", "query.spl", "dependencies.json"):
                path = rule + "/" + filename
                if path not in entries or not content(path).strip():
                    problems.append("missing_rule_file:" + path)
                elif filename.endswith(".json"):
                    data = json.loads(content(path))
                    if not isinstance(data, dict):
                        problems.append("invalid_rule_json:" + path)
                    pending.extend(
                        (prefix + p, prefix) for p in _tide_v3_macro_pointers(data)
                    )
        visited = set()
        while pending:
            path, prefix = pending.pop()
            if path in visited:
                continue
            visited.add(path)
            if path not in entries:
                problems.append("missing_macro:" + path)
                continue
            data = json.loads(content(path))
            if not isinstance(data, dict):
                problems.append("invalid_macro_json:" + path)
            pending.extend((prefix + p, prefix) for p in _tide_v3_macro_pointers(data))
        if not rule_dirs:
            problems.append("no_rule_triples")
        return {
            "status": "incomplete" if problems else "reusable",
            "rules": len(rule_dirs),
            "macro_files": sum("macros/by-stanza/" in p for p in entries),
            "problems": sorted(set(problems)),
            "assessment": "local_structure_only",
        }
    except (click.ClickException, OSError, ValueError, UnicodeError) as ex:
        return {"status": "incomplete", "error_type": type(ex).__name__}


_PROBE = "APISIX_PROBE_OK"
_TOOL_RESULT = "APISIX_TOOL_OK"
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "availability_echo",
            "description": "Return the supplied synthetic availability marker.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
]


def _stream_probe(client, model, messages, *, tools=False):
    kwargs: dict[str, Any] = {"stream_options": {"include_usage": True}}
    if tools:
        kwargs["tools"] = _TOOLS
    started = time.monotonic()
    text = []
    calls: dict[int, dict[str, Any]] = {}
    first_text = None
    chunks = 0
    finish = None
    with client.chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs
    ) as events:
        for event in events:
            chunks += 1
            for choice in event.choices:
                finish = choice.finish_reason or finish
                delta = choice.delta
                if delta.content:
                    text.append(delta.content)
                    if first_text is None:
                        first_text = time.monotonic() - started
                for item in delta.tool_calls or []:
                    call = calls.setdefault(
                        item.index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if item.id:
                        call["id"] = item.id
                    if item.function:
                        call["function"]["name"] += item.function.name or ""
                        call["function"]["arguments"] += item.function.arguments or ""
    safe_finish = (
        finish
        if finish
        in (None, "stop", "length", "tool_calls", "content_filter", "function_call")
        else "unknown"
    )
    metrics = {
        "chunks": chunks,
        "text_deltas": len(text),
        "incremental_observed": len(text) > 1,
        "first_text_seconds": first_text,
        "total_seconds": time.monotonic() - started,
        "finish_reason": safe_finish,
    }
    if tools:
        values = list(calls.values())
        valid = len(values) == 1 and bool(values[0]["id"])
        if valid:
            function = values[0]["function"]
            valid = function["name"] == "availability_echo" and json.loads(
                function["arguments"]
            ) == {"value": _TOOL_RESULT}
        valid = valid and finish == "tool_calls"
        return {"status": "passed" if valid else "failed", **metrics}, values
    valid = "".join(text).strip() == _PROBE and finish == "stop"
    status = "passed" if valid and len(text) > 1 else "failed"
    result = {"status": status, **metrics}
    if valid and len(text) == 1:
        result["reason"] = (
            "Only one text delta; incremental delivery remains unverified"
        )
    return result, []


def availability(name, profile, model):
    checks = {
        key: {"status": "not_tested"}
        for key in (
            "models",
            "completion",
            "streaming",
            "function_call",
            "tool_continuation",
        )
    }
    report = {
        "schema_version": 1,
        "profile": name,
        "model": model,
        "python": platform.python_version(),
        "tls_validation": profile.get("verify_ssl", True),
        "client_timeout": None,
        "sdk_retries": 0,
        "checks": checks,
    }
    checks["knowledge"] = inspect_knowledge(profile.get("knowledge_dir"))
    checks["splunk_mcp"] = {
        "status": "not_tested",
        "reason": "Independent workplace MCP access has not been established",
    }
    with _client(name, profile) as client:
        try:
            models = _discover(client)
            _cache_models(name, models)
            checks["models"] = {
                "status": "passed",
                "count": len(models),
                "selected_model_listed": model in models,
            }
        except _PROBE_ERRORS as ex:
            checks["models"] = _safe_error(ex)
            if checks["models"].get("http_status") in (401, 403):
                return report
        messages = [{"role": "user", "content": "Reply with exactly " + _PROBE}]
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, stream=False
            )
            valid = (
                bool(response.choices)
                and response.choices[0].message.content == _PROBE
                and response.choices[0].finish_reason == "stop"
            )
            checks["completion"] = {"status": "passed" if valid else "failed"}
        except _PROBE_ERRORS as ex:
            checks["completion"] = _safe_error(ex)
            if checks["completion"].get("http_status") in (401, 403):
                return report
        try:
            checks["streaming"], _ = _stream_probe(client, model, messages)
        except _PROBE_ERRORS as ex:
            checks["streaming"] = _safe_error(ex)
        tool_messages = [
            {
                "role": "user",
                "content": 'Call availability_echo with value "'
                + _TOOL_RESULT
                + '" then reply with exactly the returned value.',
            }
        ]
        try:
            checks["function_call"], calls = _stream_probe(
                client, model, tool_messages, tools=True
            )
            if checks["function_call"]["status"] == "passed":
                continuation = [
                    *tool_messages,
                    {"role": "assistant", "content": None, "tool_calls": calls},
                    {
                        "role": "tool",
                        "tool_call_id": calls[0]["id"],
                        "content": _TOOL_RESULT,
                    },
                ]
                response = client.chat.completions.create(
                    model=model, messages=continuation, stream=False
                )
                valid = (
                    bool(response.choices)
                    and response.choices[0].message.content == _TOOL_RESULT
                    and response.choices[0].finish_reason == "stop"
                )
                checks["tool_continuation"] = {
                    "status": "passed" if valid else "failed"
                }
        except _PROBE_ERRORS as ex:
            target = (
                "tool_continuation"
                if checks["function_call"]["status"] == "passed"
                else "function_call"
            )
            checks[target] = _safe_error(ex)
    return report


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def apisix():
        "Configure APISIX profiles and check OpenAI-compatible API availability."

    @apisix.group()
    def profile():
        "Manage local APISIX connection profiles."

    @profile.command(name="create")
    @click.argument("name")
    @click.option(
        "--base-url",
        required=True,
        prompt="OpenAI-compatible API base URL (including /v1)",
    )
    @click.option("--verify-ssl/--no-verify-ssl", default=True, show_default=True)
    def create(name, base_url, verify_ssl):
        "Create a profile without altering existing profiles or storing a key."
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise click.ClickException(
                "Use a lowercase profile name containing letters, digits and hyphens"
            )
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in ("https", "http")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise click.ClickException(
                "Provide an HTTP(S) API base URL without credentials, query or fragment"
            )
        config = _config()
        if name in config["profiles"]:
            raise click.ClickException("Profile already exists; no changes made")
        alias = "apisix-" + name
        config["profiles"][name] = {
            "base_url": base_url.rstrip("/"),
            "verify_ssl": verify_ssl,
            "key_alias": alias,
            "models": [],
        }
        _write_json(_config_path(), config)
        click.echo("Profile created. Store its key with: llm keys set " + alias)
        if not verify_ssl:
            click.echo("TLS certificate validation is disabled for this profile.")

    @apisix.command(name="models")
    @click.option("--profile", "name", required=True)
    def models(name):
        "Discover and cache actual model IDs for a profile."
        try:
            with _client(name, _profile(name)) as client:
                found = _discover(client)
            _cache_models(name, found)
        except _PROBE_ERRORS as ex:
            raise click.ClickException(
                f"Model discovery failed: {_safe_error(ex)}"
            ) from None
        for model in found:
            click.echo(f"apisix/{name}/{model}")

    @apisix.command(name="knowledge")
    @click.option("--profile", "name", required=True)
    @click.option(
        "--directory", type=click.Path(exists=True, file_okay=False), required=True
    )
    def knowledge(name, directory):
        "Assess and remember an existing local TIDE folder, without uploading."
        _profile(name)
        result = inspect_knowledge(directory)
        config = _config()
        config["profiles"][name]["knowledge_dir"] = str(Path(directory).resolve())
        _write_json(_config_path(), config)
        click.echo(json.dumps(result, indent=2))
        if result["status"] != "reusable":
            raise click.ClickException(
                "Knowledge structure requires inspection; do not recreate it yet"
            )

    @apisix.command(name="check")
    @click.option("--profile", "name", required=True)
    @click.option(
        "--model",
        required=True,
        help="Exact provider model ID from discovery, without the apisix/profile/ prefix.",
    )
    @click.option("--output", type=click.Path(dir_okay=False), required=True)
    def check(name, model, output):
        "Run synthetic model, streaming and function-call checks; write a safe report."
        report = availability(name, _profile(name), model)
        _write_json(output, report)
        click.echo(json.dumps(report, indent=2))
        required = (
            "models",
            "completion",
            "streaming",
            "function_call",
            "tool_continuation",
        )
        if any(report["checks"][key]["status"] != "passed" for key in required):
            raise click.ClickException(
                "One or more API probes failed or were not tested; see report"
            )
