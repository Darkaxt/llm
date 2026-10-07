"""Open WebUI provider for LLM.

This built-in plugin keeps Open WebUI-specific auth and transport isolated from
LLM's conversation engine. It uses openwebui-sdk for email/password sign-in,
model discovery, streaming, reasoning and the Socket.IO tool execution path.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import click
import llm
from llm.parts import AttachmentPart, ReasoningPart, StreamEvent, TextPart, ToolResultPart
from openwebui_sdk import OpenWebUIClient
from openwebui_sdk.errors import APIError, AuthError

CONFIG_FILENAME = "openwebui.json"


def _config_path() -> Path:
    return llm.user_dir() / CONFIG_FILENAME


def _load_config() -> dict[str, Any] | None:
    path = _config_path()
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _save_config(config: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def _client(config: dict[str, Any]) -> OpenWebUIClient:
    url = config.get("url")
    token = config.get("token")
    if not url or not token:
        raise llm.ModelError(
            "Open WebUI is not configured. Run: llm openwebui login --url URL --email EMAIL"
        )
    return OpenWebUIClient(base_url=str(url), token=str(token))


def _model_cache(client: OpenWebUIClient) -> list[dict[str, str]]:
    return [
        {"id": model.id, "name": model.name or model.id}
        for model in client.list_models()
        if model.id
    ]


def _persist_session(config: dict[str, Any], client: OpenWebUIClient) -> dict[str, Any]:
    session = client.session()
    config = dict(config)
    config["token"] = session.token
    config["user"] = {
        "id": session.user_id,
        "email": session.email,
        "name": session.name,
        "role": session.role,
    }
    _save_config(config)
    return config


def _message_content(message: Any) -> str:
    chunks: list[str] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            chunks.append(part.text)
        elif isinstance(part, ToolResultPart):
            chunks.append(part.output)
        elif isinstance(part, ReasoningPart):
            # Provider-side reasoning is not part of the visible conversation
            # context and should not be sent back as ordinary text.
            continue
        elif isinstance(part, AttachmentPart):
            raise llm.ModelError(
                "Open WebUI attachments are not implemented yet in this provider"
            )
    return "\n".join(chunk for chunk in chunks if chunk)


def _messages_for_openwebui(prompt: llm.Prompt) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in prompt.messages:
        content = _message_content(message)
        if content:
            messages.append({"role": message.role, "content": content})
    return messages


class OpenWebUIModel(llm.Model):
    """One model exposed by the configured Open WebUI instance."""

    class Options(llm.Options):
        temperature: float | None = None
        openwebui_tools: bool = True

    def __init__(self, remote_model_id: str, display_name: str | None = None):
        self.remote_model_id = remote_model_id
        self.display_name = display_name or remote_model_id
        self.model_id = f"openwebui/{remote_model_id}"

    def __str__(self) -> str:
        return f"Open WebUI: {self.display_name}"

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None,
    ) -> Iterator[str | StreamEvent]:
        config = _load_config()
        if not config:
            raise llm.ModelError(
                "Open WebUI is not configured. Run: llm openwebui login --url URL --email EMAIL"
            )

        client = _client(config)
        messages = _messages_for_openwebui(prompt)

        try:
            tool_ids = client.resolve_tools(
                self.remote_model_id,
                no_tools=not prompt.options.openwebui_tools,
            )
        except (APIError, AuthError) as exc:
            raise llm.ModelError(str(exc)) from exc

        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        tool_activity: list[str] = []
        status_activity: list[str] = []

        def on_text(fragment: str) -> None:
            events.put(("text", fragment))

        def on_reasoning(fragment: str) -> None:
            events.put(("reasoning", fragment))

        def on_tool(line: str) -> None:
            tool_activity.append(line)

        def on_status(line: str) -> None:
            status_activity.append(line)

        def worker() -> None:
            try:
                result = client.run_chat(
                    model=self.remote_model_id,
                    messages=messages,
                    tool_ids=tool_ids,
                    temperature=prompt.options.temperature,
                    on_text=on_text,
                    on_reasoning=on_reasoning,
                    on_tool=on_tool,
                    on_status=on_status,
                )
                events.put(("done", result))
            except Exception as exc:  # handed back to the main iterator
                events.put(("error", exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            kind, payload = events.get()
            if kind == "text":
                yield str(payload)
            elif kind == "reasoning":
                if not prompt.hide_reasoning:
                    yield StreamEvent(type="reasoning", chunk=str(payload))
            elif kind == "error":
                if isinstance(payload, (APIError, AuthError)):
                    raise llm.ModelError(str(payload)) from payload
                raise payload
            elif kind == "done":
                result = payload
                response.response_json = {
                    "provider": "openwebui",
                    "remote_model": self.remote_model_id,
                    "reasoning": result.reasoning,
                    "tool_calls": result.tool_calls,
                    "tool_activity": tool_activity,
                    "status_activity": status_activity,
                }
                break


@llm.hookimpl
def register_models(register):
    config = _load_config()
    if not config:
        return
    for item in config.get("models", []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        remote_id = str(item["id"])
        model = OpenWebUIModel(remote_id, item.get("name"))
        register(model, aliases=[f"owui/{remote_id}"])


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="openwebui")
    def openwebui_group():
        """Configure and inspect the Open WebUI provider."""

    @openwebui_group.command(name="login")
    @click.option("--url", required=True, help="Open WebUI base URL")
    @click.option("--email", required=True, help="Open WebUI email address")
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=False)
    def login(url: str, email: str, password: str):
        """Sign in with email/password and cache the visible model catalogue."""
        client = OpenWebUIClient(base_url=url)
        try:
            session = client.signin(email, password)
            models = _model_cache(client)
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc

        config = {
            "url": url.rstrip("/"),
            "email": email,
            "token": session.token,
            "user": {
                "id": session.user_id,
                "email": session.email,
                "name": session.name,
                "role": session.role,
            },
            "models": models,
        }
        _save_config(config)
        click.echo(
            f"Saved Open WebUI session for {session.email or email}; "
            f"{len(models)} model(s) cached."
        )

    @openwebui_group.command(name="whoami")
    def whoami():
        """Validate the current session and show its identity."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            config = _persist_session(config, client)
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(config.get("user", {}), indent=2, ensure_ascii=False))

    @openwebui_group.command(name="sync")
    def sync():
        """Refresh the session and cached model catalogue."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        client = _client(config)
        try:
            config = _persist_session(config, client)
            config["models"] = _model_cache(client)
            _save_config(config)
        except (APIError, AuthError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Cached {len(config['models'])} Open WebUI model(s).")

    @openwebui_group.command(name="models")
    def models():
        """Show the cached Open WebUI model catalogue."""
        config = _load_config()
        if not config:
            raise click.ClickException("Open WebUI is not configured")
        for item in config.get("models", []):
            click.echo(
                f"openwebui/{item.get('id')}\t{item.get('name') or item.get('id')}"
            )
