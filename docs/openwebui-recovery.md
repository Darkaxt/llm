# Open WebUI MCP checkpoint recovery

The Open WebUI provider checkpoints native server-side MCP tool calls while
using Socket.IO, without modifying the server-side model. This is separate
from the one-shot background_legacy tool prepass.

## Start an investigation

~~~powershell
..\.venv\Scripts\python.exe -m llm chat -m openwebui/glm-5.3 -o openwebui_tools true -o openwebui_mcp_transport background_native
~~~

Record the conversation ID printed by the CLI.

Interactive terminal chat uses warm orange (`#d97757`) for program messages,
turn separators, the prompt marker and the Open WebUI status line. Your input
and **You** label are blue; replies and the **LLM** label are ivory. Reasoning
uses a separate gray foreground rather than terminal dimming. The terminal's
background is retained. Redirected output and saved journals remain plain;
set `NO_COLOR` to disable this chat palette. Recovery uses the same reply,
reasoning and status colors in an interactive terminal.

Tool activity remains inline in muted dark pink (`#c4779e`), so later status
updates do not erase it. After a native result is durably checkpointed, its
notice includes the actual tool name, call ID, full arguments and the saved
result's relative path, byte count and SHA-256. A returned result is not a
claim that its contents indicate success; inspect the checkpoint for the full
output. Final arguments are shown with that returned result, not as an approval
step before execution. Start/done notices remain visible as received.

## Live HTML conversation report

Each chat automatically maintains `<conversation-id>.html` beside its JSONL
journal, including when `LLM_CHAT_EXPORT_DIR` changes the conversations folder.
The CLI prints `Autosave HTML:` at startup. Open that file in a browser and
refresh it while generation runs to see the latest saved text. It is a single
offline document, with no server, scripts or external assets required.

The report preserves input, replies, visible reasoning, inline tool notices
and program messages in the terminal palette. Its footer shows current status;
streamed chunks join without per-token line breaks. `--hide-reasoning` also
hides reasoning in the report for that session. The dark report palette remains
available when terminal colors are disabled using `NO_COLOR`.

The full visible conversation, reasoning and tool arguments may contain incident
data. Review them before sharing the HTML with your team. Hidden provider request
metadata and raw result sidecars are not embedded. Keep matching checkpoints
separately when your team needs to inspect complete returned results.

Reports update incrementally without rewriting all earlier text for each chunk.
Reopening a journal in a fresh process rebuilds its report and continues the
same file. Original JSONL/checkpoints remain the recovery source of truth. A
report I/O error prints a notice and disables report updates for that session;
journaling and inference continue. Reopening can rebuild after the filesystem
problem is resolved. A forcibly interrupted report write can likewise be rebuilt
from the journal on the next open.

## Stored evidence

On every **completed** native tool call, the fork stores the exact tool name,
call ID, arguments, and full JSON result. The sidecar file is written and
synced before its pointer is written and synced in the JSONL journal.

The default storage locations on Windows are:

~~~text
%APPDATA%\io.datasette.llm\conversations\<conversation-id>.jsonl
%APPDATA%\io.datasette.llm\conversations\<conversation-id>.html
%APPDATA%\io.datasette.llm\conversations\<conversation-id>.checkpoints\<run-hash>\<call-hash>.json
~~~

LLM_CHAT_EXPORT_DIR overrides the directory. Each checkpoint pointer has a
SHA-256 digest and byte count; recovery rejects missing or corrupt payloads.
A single result exceeding 32 MiB triggers an explicit error rather than
silently discarding data.

These files contain **unencrypted incident-investigation evidence**.
Protect them with the user's filesystem permissions.

## Resume a failed investigation

The old remote task is NOT restarted or reconnected. A new model request is
constructed from the previously saved TIDE context and completed tool calls,
including their actual results.

Verify the saved evidence without contacting Open WebUI:

~~~powershell
..\.venv\Scripts\python.exe -m llm openwebui resume <conversation-id> --dry-run
~~~

Recover in evidence-only mode (no tool execution), the default:

~~~powershell
..\.venv\Scripts\python.exe -m llm openwebui resume <conversation-id>
~~~

Explicitly allow the model to continue with additional MCP searches:

~~~powershell
..\.venv\Scripts\python.exe -m llm openwebui resume <conversation-id> --allow-new-searches
~~~

Only MCP servers selected in both the old request and the current CLI
configuration are used. Because the server/model controls native tool
selection, **a repeated search cannot be prevented deterministically** with
this opt-in. Use the evidence-only default for a no-search continuation.

A forcibly killed process might not log a provider_error. After verifying
that the original remote task is no longer executing, opt in to recovery of
that interrupted journal:

~~~powershell
..\.venv\Scripts\python.exe -m llm openwebui resume <conversation-id> --assume-stopped --dry-run
~~~

Use --provider-run-id to select a specific older failed provider request.
Every continuation writes a new conversation journal. The source journal
records a link to the new conversation.

Within interactive chat, !retry now redirects Open WebUI failures to
checkpoint recovery (or refuses if evidence cannot be validated). Use
!retry-force ONLY when intentionally rerunning the entire original prompt.

## Scope and limitations

The client keeps live reasoning fragments and cumulative reasoning snapshots in
stable slots throughout a request. Tool-only, empty or partial snapshots do not
erase earlier reasoning or invalidate the next continuation's stream slot.
Repeated snapshots append only new text, while completed tool checkpoints remain
independent. This fixes a reproducible `error handling chat event: list index out
of range` at a reasoning/tool boundary; it does not change model reasoning,
timeouts or retries, and is not proof of corporate five-minute endurance.

- This is **continuation from evidence**, not restoration of the provider's
  live task or of model-private reasoning.
- An incomplete tool call is not presented as successful evidence.
- Local checkpoint digests detect corruption, not malicious replacement of
  both the sidecar and its journal pointer.
- Journal appends repair an incomplete final JSONL line from a process crash.
- Prior journals without native tool-result sidecars are **not recoverable**
  from tool-status strings alone. This includes the initial six-search
  GuardDuty investigation made before checkpoint support.
- Evidence-only recovery disables newly executing tools but still requires
  the model's output to be critically reviewed.
- This patch does NOT change the corporate Open WebUI upstream timeout.
- Some endpoints can still fail during inference; future completed tool
  results remain separately recorded when native searches are enabled.

## Local offline tests

~~~powershell
..\.venv\Scripts\python.exe -m pytest -q tests/test_openwebui.py tests/test_openwebui_recovery.py tests/test_chat.py
~~~
