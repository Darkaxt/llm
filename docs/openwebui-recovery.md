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

## Readable tool results

Tool results are formatted for readability in both the terminal preview and the
HTML Full result panel. Complete JSON object/array strings are expanded and
indented, including JSON encoded inside a tool wrapper's text fields. Wrapper
fields, order, error fields and Unicode remain visible. This is a display view:
an expanded JSON string can appear as an object, while the checkpoint retains
the original string type/content. Ordinary/scalar text, malformed JSON,
duplicate-key documents and JSON numbers that would lose precision stay intact.
The model/recovery payload, sidecar bytes and digest do not change. Terminal
preview limits still apply; HTML contains the full formatted result. Rebuilding
an existing report applies the same formatting without rerunning tools.

## HTML line and event timing

Each logical conversation line has its originating journal timestamp in **UTC
with milliseconds** and a **Δ gap in seconds** from the preceding rendered event,
including status updates. The timestamp is when this client recorded the event,
not when the model started computing. Several lines from one event share the
same timestamp and gap; soft wrapping does not create another event.

Streamed words remain joined in the default conversation view. Check
**Individual events** near the top of the HTML to inspect each streamed chunk
and intermediate status with its own time/gap. This script-free checkbox works
with mouse or keyboard Space and does not change the saved conversation.
The tool metadata and full-result panels remain independently collapsible.

Gaps use the journal's wall-clock timestamps. They are observations, not internal
reasoning durations or proof of model inactivity. Missing/invalid times and
backwards clock changes are labelled explicitly rather than guessed. Hidden
reasoning and private request metadata stay excluded. The footer includes the
current status's timing.

New reports receive timing automatically. Rebuilding an existing report uses its
original JSONL timestamps and verified result sidecars, without inference or tool
execution. Keep those files together; reopening the conversation rebuilds its
HTML, or call `llm.chat_html.ensure_html_report(journal_path)` for an offline-only
rebuild. No timestamp is reconstructed from model narration.

## Cancellation and Windows idle sleep

During generation, **Ctrl+C** (or Windows Ctrl+Break) interrupts this console's
request. Escape is no longer polled across the desktop; using it in another app
cannot cancel this response through the old global-key hook. Composer bindings
are unchanged. Locking the workstation is not a cancellation command.

The terminal and HTML identify an observed process signal, such as SIGINT, or a
local KeyboardInterrupt whose origin is unknown. A process signal does not prove
a physical keypress: software can also send signals. The journal's
`turn_cancelled` record includes `source`, `signal`, `reason` and available Windows
clock samples. `provider_stop_requested` separately records local response-iterator
closure. The worker's generic stop error says **stopped by local client**, rather
than assuming the user requested it. An explicit server `task-cancelled` event is
retained as `source=server_event`, with its event name and actual server reason
(or an explicit missing reason), in provider/turn errors. Server failures are not
silently ignored or automatically retried.

Windows Open WebUI requests prevent **idle system sleep** by default, only while
the active request thread owns its execution-state request. Completion, error or
cancellation restores the same thread's prior state; the CLI waits for its local
worker to exit before returning to the next prompt. That local exit is not
independent proof that a remote server task has stopped. Native power API failure
is journaled/reported and inference continues.

Opt out for a request with `-o openwebui_keep_awake false`. Recovery has
`--keep-awake` (default) and `--no-keep-awake`. Non-Windows hosts record this
mechanism as unsupported. No power plan, display-on requirement, lock policy,
scheduled task or background service is installed. Explicit Sleep, a lid/power
action or corporate policy can still suspend the machine.

`provider_power` records acquisition/release/failure plus `tick_count_ms`
(sleep-inclusive uptime) and `awake_time_100ns` (awake-only uptime) when Windows
provides them. Cancellation captures the same raw clocks. These samples support
comparison across a request; they are not lock notifications or proof that a
sleep transition caused cancellation. Preserve OS power/session events when a
workplace lock/suspend failure needs correlation.

Windows API contracts: [execution state](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate),
[sleep-inclusive uptime](https://learn.microsoft.com/windows/win32/sysinfo/windows-time)
and [awake-only uptime](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryunbiasedinterrupttime).

Tool activity remains inline in muted dark pink (`#c4779e`), so later status
updates do not erase it. After a native result is durably checkpointed, its
notice includes the actual tool name, call ID, full arguments and the saved
result's relative path, byte count and SHA-256. A returned result is not a
claim that its contents indicate success. A result preview follows, limited to
2000 characters and 20 lines; a truncation marker points to the full HTML result
and checkpoint. These are display limits only. Final arguments are shown with
that returned result, not as an approval step before execution. Start/done
notices remain visible as received.
Transient status redraws wait while an interactive stream has an unfinished
line, so heartbeats cannot split a streamed word or sentence. Status remains
recorded in the journal and HTML footer; tools still start on their own line.

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

Each completed native tool call includes its **full returned output** in a
collapsed-by-default **Full result** panel. Click the summary (or focus it and
press Enter/Space) to expand it. A separate collapsed **Call metadata** panel
contains the call ID, full arguments, checkpoint path, size and digest.
Structured results retain every field; strings remain intact. No scripts or
sidecar access are needed to read a generated HTML
file. The complete conversation, reasoning, arguments and returned outputs may
contain incident data; review them before sharing. Hidden provider request
metadata is not embedded. Raw sidecars remain the authoritative evidence, and
must stay beside the journal for rebuilding full result panels. Missing/corrupt
sidecars produce the same report error as other failed rebuilds; inference and
journaling continue. Reopening also upgrades the previous client's completed-call
notices at their original display positions when matching checkpoint files are
present and verified. Older activity without a completed-call notice/result
checkpoint remains a notice; it cannot supply a full returned output.

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

## Tool-evidence recovery extension specification (2026-10-14)

The user authorizes expanding recovery to include tools that ran in the previous
attempt. This section is authoritative for that extension; reasoning remains
excluded. Earlier failed-turn recovery already reconstructs verified tool calls.

- P1: Preserve every completed checkpoint's call ID, tool name, arguments and
  full result, including KB builtins and MCP tools, without executing it again.
  Verify sidecar identity, size and digest; reject missing/conflicting evidence.
- P2: Add an explicit `--empty-answer` option for a terminal completed turn whose
  answer is demonstrably empty and which has completed tool evidence. Refuse
  nonempty/unknown completed answers and active runs. Existing failed-turn and
  confirmed process-killed recovery remain available. Bound selection to the
  chosen provider run, so a later turn cannot invalidate an older failed attempt.
- P3: Copy inherited verified checkpoints into each new recovery journal with
  source provenance, distinguishing inherited calls from execution in that run.
  A subsequent failed recovery must preserve those calls plus any newly completed
  calls, exactly once in model history, without relying on the original files.
- P4: Keep evidence-only recovery as default; retain explicit original/current
  MCP scope for new searches. Exclude reasoning, preserve original source files,
  and introduce no automatic retries, deadlines, model/provider changes or tools.
- P5: Verify complete CLI dry-run and continuation flows, including chained
  failure/recovery, empty completion, run selection, provenance and corrupt
  evidence rejection. Document usage, commit/push and deliver commands in issue
  #28. Offline fixtures do not certify corporate recovery.

| Stage | Status | Reconciliation |
|---|---|---|
| Previous-attempt tool evidence recovery (P1-P5) | COMPLETE | P1-P5 satisfied: verified local flows, committed/pushed implementation and issue delivery; no blockers or deferrals |

Eligibility/run selection and inherited evidence blockers are resolved. Complete
offline CLI flows verify empty-answer eligibility, active/nonempty/tool-free
refusal, KB/MCP preservation, source provenance in terminal/HTML, chained recovery
after original sidecar removal, no duplicate history, copy failure and corruption
rejection. A recovery with an empty result fails explicitly while retaining its
checkpoints. Affected provider/chat/HTML/timing/power regressions pass with the
existing expected-failure/XPASS outcomes retained. Black, Ruff, focused source
typing, generated CLI documentation and whitespace checks pass. An optional
untyped-body check reproduces the same three Click Path typing errors in unchanged
knowledge commands at the pre-change HEAD; these are outside this extension.

Implementation committed/pushed and remote-verified as `b78ad024224dc7947fe600ae9febaf62f5630298`.
[Issue #28 delivery](https://github.com/Darkaxt/XSoar-MDR/issues/28#issuecomment-6178193130)
contains update, dry-run and recovery commands. P1-P5 have no remaining internal
advancement blockers or required deferrals. Closure review found no
requirement-blocking design debt. Corporate endurance remains the separate
external issue #28 validation boundary and is not certified by these local flows.

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

Completed turns that produced no answer can be recovered explicitly, provided
they contain at least one verified completed tool checkpoint:

~~~powershell
..\.venv\Scripts\python.exe -m llm openwebui resume <conversation-id> --empty-answer --dry-run
..\.venv\Scripts\python.exe -m llm openwebui resume <conversation-id> --empty-answer
~~~

`--empty-answer` accepts only a recorded empty/whitespace-only answer. A missing
answer field does not prove emptiness; successful nonempty turns and active runs
remain protected. Use `--provider-run-id` for an older attempt, even when later
turns completed. No reasoning is added to the recovery request, and a reasoning-
only turn with no completed tools still has no recoverable tool evidence.

Each recovery copies all verified completed calls into its own sidecars,
including knowledge-base builtins as well as MCP calls. `inherited_from` records
the source conversation, provider run and call ID; `executed_in_this_run: false`
distinguishes these copies from newly executed calls. Full results are visible in
the new HTML report. Existing source files are preserved. A failed recovery can
itself be recovered using its new conversation ID, retaining both inherited
results and any new completed calls without duplicating them in model history.
Its own copies are sufficient; the original sidecars are not needed again.
Conflicting history, corrupt/missing sidecars and incomplete inheritance fail
explicitly. A recovery that returns no final answer is recorded as an error,
with its tool evidence retained for another explicit recovery.

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
