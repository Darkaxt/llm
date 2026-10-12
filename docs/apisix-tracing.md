# APISIX reasoning and request timing capture

Authoritative requirements, workplace commands and artifact replies:
[XSoar-MDR #28](https://github.com/Darkaxt/XSoar-MDR/issues/28).
This implements that issue's C1-C4 diagnostic follow-up. The workplace endurance
run and platform-owner handoff remain separate, externally verified stages.

## Measure a run that has already completed

The existing response log contains `duration_ms`, measured with the client's
monotonic clock around response execution. No new capture or rerun is needed to
recover that duration. Select the exact completed conversation:

```powershell
$LlmPython = (Resolve-Path "..\.venv\Scripts\python.exe").Path
$ConversationId = Read-Host "Completed test conversation ID"
$LogJson = & $LlmPython -m llm logs list --cid $ConversationId --json -n 0
if ($LASTEXITCODE -ne 0) { throw "Conversation log export failed" }
$Turns = $LogJson | ConvertFrom-Json
$Turns | Select-Object id, model, datetime_utc, duration_ms, @{Name='duration_seconds';Expression={if ($null -ne $_.duration_ms) { $_.duration_ms / 1000.0 }}}, input_tokens, output_tokens | Format-Table -AutoSize
```

This is elapsed response execution, including waiting and output consumption,
not internal reasoning time. Separate turns have separate durations. A model's
answer about how long it took is a different observation and is not a clock.
Existing APISIX logs cannot recover reasoning fields previously discarded by the
adapter. Preserve the original journal and completed conversation.

## Capture subsequent APISIX requests

Capture is disabled by default. Enable it for the current PowerShell process,
using an approved local directory, then run the existing synthetic workload:

```powershell
$LlmPython = (Resolve-Path "..\.venv\Scripts\python.exe").Path
$PreviousTraceDirectory = $env:LLM_APISIX_TRACE_DIR
$TraceDirectory = Join-Path $PWD ("apisix-trace-" + [guid]::NewGuid().ToString('N'))
$env:LLM_APISIX_TRACE_DIR = $TraceDirectory
try {
    & $LlmPython -m llm chat -m "apisix/work/glm-5.3"
} finally {
    if ($null -eq $PreviousTraceDirectory) {
        Remove-Item Env:\LLM_APISIX_TRACE_DIR -ErrorAction SilentlyContinue
    } else {
        $env:LLM_APISIX_TRACE_DIR = $PreviousTraceDirectory
    }
}
```

The previous process-local capture setting is restored afterward. This changes
neither the saved profile/key nor other providers. Capture introduces no client deadline,
retry, artificial delay, token cap or reasoning-effort override. It does not add
visible reasoning or a live timer to the chat renderer.

The selected directory contains one `apisix-<trace_id>.jsonl` per HTTP request.
Each records UTC start, monotonic elapsed seconds, request body, response status,
content type/encoding, optional `X-Request-ID`, and exact ordered response-body
chunks in base64. Response chunks are captured before HTTP content decompression
and SDK parsing; comments, unknown JSON fields and reasoning fields are retained.
This is an HTTP body trace, not a TLS packet capture or browser HAR. HTTP transfer
framing has already been removed by the transport.

URLs, Authorization/Cookie/Set-Cookie headers and exception messages are omitted
from metadata. **Bodies are complete and unredacted**, including growing prompt
history, reasoning and tool results if present. Base64 is not encryption. Keep
raw traces private at work and use synthetic content. Capture I/O failures are
reported rather than silently continuing without the requested evidence.

## Summarize a trace offline

```powershell
& $LlmPython -m llm apisix trace-summary "FULL_PATH_TO_ONE_TRACE.jsonl"
```

The command makes no network requests and does not print prompt, response or
reasoning text. It replays HTTP decompression and reconstructs SSE frames using
the last captured chunk required to complete each frame. It reports:

- Request start and observed duration; unfinished traces have no final duration.
- First byte/event, first reasoning and first answer arrival.
- Maximum no-byte gap, including initial waiting and the final interval before
  observed termination; maximum gap between reasoning events.
- Reasoning/answer/tool event counts and first-to-last reasoning event span.
- Provider finish reasons and numeric token usage when supplied.
- `[DONE]`, EOF, incomplete SSE/JSONL tails, malformed events, provider errors,
  decompression failures and observed transport error/cancellation type.

Timing includes consumer backpressure and capture/output work; it is not a
server-side measurement. Multiple events in one chunk have the same observed
delivery time. A 308-second reasoning-event span can include a long silent gap;
it does not prove 308 seconds of continuous reasoning or internal computation.
A single reasoning event has a zero observed span, not zero computation time.

`outcome` describes the HTTP stream, not the complete application outcome. The
SDK normally closes a response after `[DONE]`, so `closed` with no observed EOF
can be normal. `closed` alone is ambiguous. A KeyboardInterrupt during transport
iteration is recorded as `cancelled`; cancellation or parser errors elsewhere
also require the chat journal/console evidence. No finish reason, HTTP 200,
EOF or `[DONE]` alone certifies a complete answer. Check them together with the
required terminal marker and actual content. The capture does not drain unread
upstream data after the SDK closes a response.

Review summaries before posting them in #28; identifiers and usage metadata can
still require redaction. Keep raw trace and the matching conversation journal
locally unless a reviewed, sanitized export is specifically needed. Capture can
observe only reasoning actually returned by the provider.
