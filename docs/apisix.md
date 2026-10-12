# APISIX workplace setup and availability checks

Job and artifact replies: [XSoar-MDR issue #28](https://github.com/Darkaxt/XSoar-MDR/issues/28).
Implementation requirements and stage ledger: [APISIX specification](apisix-spec.md).

The APISIX provider uses the OpenAI-compatible Chat Completions API and the
existing LLM conversation and function-tool loop. It does not use Open WebUI's
server-side knowledge or MCP APIs. Keep the existing Open WebUI profile and KB.

## Install the review branch

Run inside your existing LLM source checkout on the work computer. Preserve any
local changes; Git will refuse a conflicting switch rather than discard them.
Select the Python executable from the environment where LLM is installed. The
example below uses the existing workplace `Projects\.venv` beside the checkout;
adjust that one path if your environment is elsewhere. Use the same executable
for installation and every subsequent command. A bare `python` may select a
different installation even when you are inside the source checkout.

```powershell
git fetch origin
git switch codex/apisix-profile-availability
git pull --ff-only
$LlmPython = (Resolve-Path "..\.venv\Scripts\python.exe").Path
& $LlmPython -m pip install -e .
git rev-parse HEAD
& $LlmPython --version
& $LlmPython -c "import sys, sqlite_utils, llm; print(sys.executable); print(llm.__file__)"
& $LlmPython -m llm apisix --help
```

Git can automatically track the matching remote branch on the first switch.
If your origin is not `Darkaxt/llm`, fetch that fork through a separate remote
instead of replacing the existing origin.

## Create the first profile

```powershell
& $LlmPython -m llm apisix profile create work --no-verify-ssl
& $LlmPython -m llm keys set apisix-work
& $LlmPython -m llm apisix models --profile work
```

At the first prompt, enter the complete `OPENAI_BASE_URL` from your email,
including `/v1`. At the hidden key prompt, paste the real gateway key. Do not
paste the masked example from the issue. The profile stores a key alias;
the existing LLM key store holds the key. Alternatively, set the process-local
`LLM_APISIX_WORK_API_KEY` environment variable through your approved credential
workflow. The generic `OPENAI_API_KEY` is not used for this profile.

Certificate validation is enabled by default. The explicit `--no-verify-ssl`
above follows the access instructions supplied in issue #28 and applies only
to this profile. Reports disclose `tls_validation: false`. Neither OS trust
nor other providers' certificate settings change. Redirect following is disabled.

The profile refuses duplicate creation. Model discovery lists only IDs actually
returned by `/models`, prefixed with `apisix/work/` for LLM model selection.
No network request occurs when ordinary `llm models` reads the cached IDs.
An unsupported `/models` endpoint does not prove inference is unavailable:
report that failure and obtain a real model ID from the service owner before
running model probes. Do not guess an ID.

## Assess the existing TIDE bundle

```powershell
& $LlmPython -m llm apisix knowledge --profile work --directory "D:\path\to\existing-TIDE-folder"
```

Use the existing local folder containing the instructions and rule bundle.
ZIP and 7z files inside that folder are read through the existing virtual-folder
reader without extraction to disk. The check requires nonempty `SKILL.md` and
`SPEC.md`, complete `detection.json`/`query.spl`/`dependencies.json` rule triples,
readable JSON objects, and resolvable macro pointers. Dependencies marked `skip`
or `inspect_if_needed` retain their existing policy.

`reusable` means the checked local structure is usable; it does not prove all
rule semantics, local prompt injection, or remote KB availability. `incomplete`
means inspect the reported missing files or JSON error before rebuilding.
`not_configured` in the availability report means no local folder has been
selected. No KB is uploaded, recreated, deleted, or modified. APISIX knowledge
injection and direct Splunk MCP integration belong to the later investigation
stage after workplace access has been established.

## Probe a discovered model

Replace `EXACT_PROVIDER_MODEL_ID` with a returned model ID, removing only the
`apisix/work/` prefix. Preserve slashes within the actual provider ID.

```powershell
& $LlmPython -m llm apisix check --profile work --model "EXACT_PROVIDER_MODEL_ID" --output apisix-availability.json
```

The probes run sequentially: model discovery, ordinary completion, SSE text,
streamed function-call arguments, and continuation after a synthetic local tool
result. They send synthetic markers only and execute no Splunk searches. The
tool probe uses the same automatic function-selection behavior as the LLM client.

The report records each outcome, HTTP error status when available, stream event
counts, first-text and total timing, TLS policy, and the local knowledge summary.
`streaming` passes only when the expected complete response and terminal finish
are observed with multiple text deltas. A single delta leaves incremental
delivery unverified. Multiple received deltas do not alone prove the gateway
flushes them promptly; review first-text/total timing, then validate the actual
long investigation stream in the workplace stage.

No client deadline, SDK retry, token cap, temperature, or reasoning-effort override
is introduced. A pending request remains pending until the provider finishes or
you cancel it. SDK/server errors are summarized by type and status, without raw
response bodies, URLs or request headers. An authentication rejection stops
dependent probes. Failed or untested API checks produce a nonzero exit code;
the report is still written for handled API/protocol failures. Knowledge and MCP
outcomes remain separate from the API probe exit status.

After passing synthetic probes, ordinary client commands are available:

```powershell
& $LlmPython -m llm -m "apisix/work/EXACT_PROVIDER_MODEL_ID" "Reply with APISIX_PROBE_OK"
& $LlmPython -m llm chat -m "apisix/work/EXACT_PROVIDER_MODEL_ID"
```

These commands use provider reasoning defaults unless you explicitly supply an
option. Client function tools use the existing `--tool` workflow. Their actual
availability is established by the report, not by the model's name.

## Report the first workplace artifact

In issue #28, attach `apisix-availability.json` and include the exact commit,
Python version, discovered model ID, command, and expected versus observed outcome.
If discovery fails before a model is known, paste its sanitized error type/status
and stop there. Include the knowledge summary if it has been run. Inspect the
artifact before posting and keep keys, headers, local configuration and the
original email off GitHub.

For direct Splunk MCP, report only whether you already have an approved endpoint,
the advertised transport (`streamable HTTP`, legacy SSE, or stdio), and the
authentication method's name. Do not post its address or credential. Without
that contract, the report explicitly marks MCP `not_tested`; neither the APISIX
gateway nor the inaccessible ITSEC metrics service establishes MCP access.

Each issue reply will identify passed/remaining criteria, evidence and root cause,
implemented fixes with commits and verification, and exact next commands. Real
corporate availability and the complete investigation remain required before the
migration can be declared complete.
