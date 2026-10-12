# APISIX profile, availability and timing specification

Authoritative job and artifact coordination: [XSoar-MDR #28](https://github.com/Darkaxt/XSoar-MDR/issues/28).
Authorization: already authorized by the user's request to merge Open WebUI,
focus on APISIX, create the first client profile, assess KB reuse, and run
availability tests. Corporate infrastructure is inaccessible here; credentials
remain on the work computer.

The issue's current R1-R7 and C1-C4 requirements remain authoritative. The owner
now handles Open WebUI integration; the remaining workplace validation concerns
long-running requests and conversations. Standalone KB/MCP integration was
removed from acceptance by the user's explicit scope change.

## Acceptance criteria for stage 2

- A1 (R2): Create named profiles from a user-supplied API base URL, store a key
  alias rather than a secret, refuse accidental replacement, and register only
  explicitly discovered models. Existing Open WebUI state is preserved.
  The user's issue comment of 2026-10-12 confirms an OpenAI-compatible `/v1`
  endpoint and currently unavailable certificate validation. Support an explicit
  per-profile TLS verification opt-out, retain verification by default, and
  disclose the selected policy in reports. The ITSEC metrics service is unavailable
  and is not part of the inference availability checks.
- A2 (R3): Generate a report covering model discovery/authentication, ordinary
  completion, incremental streaming, streamed function calls, and continuation
  with a locally supplied synthetic tool result. The report must distinguish
  failed and untested probes and must not include credentials, corporate URLs,
  absolute KB paths, or incident content. No Splunk calls occur in these probes.
- A3 (historical structural probe): Assess the existing local TIDE v0.3 folder or archive using the
  existing virtual-folder reader. Verify instructions, complete rule triples,
  JSON readability and macro references. Report reusable/incomplete/not
  configured without uploading, deleting, rebuilding or altering knowledge.
  This check does not establish KB content injection or MCP integration.
- A4 (R3): Reuse the existing Chat Completions implementation for interactive
  chat and client function tools. Preserve provider reasoning defaults. APISIX
  requests have no client deadline or automatic SDK retries. Diagnostics must
  report actual function calling and streaming behavior, not infer support from
  model names or HTTP 200 alone.
- A5 (R6): Publish exact setup and diagnostic commands, expected artifacts,
  fixes and verification evidence in issue #28. Only synthetic probes execute
  in this development environment. Required workplace availability remains a
  stage 3 acceptance criterion.

## Stage ledger

This is the local verification snapshot before the capture commit and ticket
delivery. Issue #28 maintains the current status after those actions.

| Stage | Status | Evidence / remaining criteria |
|---|---|---|
| 1. Preserve Open WebUI baseline | COMPLETE | PR #1 merged as `31b9bce`; regression/dependency, lint, formatting, type, documentation and packaging QA inspected |
| 2. Profile and availability diagnostics | COMPLETE | A1-A4 verified with protocol/client-loop fixtures and affected regressions; A5 setup/report instructions delivered in issue #28 |
| 3. Workplace endurance validation | BLOCKED | R4/R5: user reports one task completed without crashing, but measured duration, multi-turn/reopen and cancellation/retry artifacts remain external requirements |
| Capture implementation follow-up | ACTIVE | C1-C3 verified: raw HTTP bodies, monotonic timings and offline summary; C4 commit and ticket delivery remain |
| 4. Findings and platform-owner handoff | NOT STARTED | R6/R7: reconcile actual workplace artifacts against E1-E5 and report observed bounds |

## Capture follow-up acceptance

- C1: Opt-in APISIX request/response body capture before SDK parsing, preserving
  exact ordered HTTP body bytes, UTC start and monotonic chunk times. Omit gateway
  URLs, credential/cookie headers and exception messages from metadata; raw
  payloads stay private until reviewed.
- C2: Offline summary measures request duration, first byte/reasoning/answer,
  byte/event gaps, observed reasoning-event span, finish reasons and usage,
  `[DONE]`, EOF, errors and cancellation. Observed delivery is not computation
  time. Unknown fields and compressed body bytes remain in the raw artifact.
- C3: No model/reasoning override, deadline, retry or other-provider change.
  Verify the actual SDK path with reasoning-only, fragmented, compressed,
  truncated and cancelled synthetic streams. A simulated clock verifies timing
  calculations; it is not live endurance evidence.
- C4: Publish the verified commit, workplace capture commands, existing-run
  duration extraction and artifact instructions in issue #28.

Implementation and verification execute as one capture stage, followed by commit
and issue delivery. No other implementation stage is ACTIVE. No required capture
work is deferred. Corporate duration and live capture evidence remain an external
blocker for Stage 3, not a condition local fixtures can satisfy.

Capture verification: the actual SDK and shared CLI retain answer output while
raw traces preserve both reasoning fields, heartbeats, unknown fields and gzip
bytes. Offline summaries handle fragmented/multiline/BOM SSE, truncation,
incomplete JSONL, corrupt compression, connection/read errors and cancellation.
A simulated 308-second reasoning-event span and 316-second request verify the
clock calculations without sleeping. CLI conversation log export independently
exposes the existing measured `duration_ms`. APISIX availability and shared chat
regressions pass with the pre-existing expected-failure/XPASS outcomes retained.
Ruff, Black, mypy, Cog and diff whitespace checks are the final verification gates.

## Historical availability evidence

The records below document the earlier availability/KB investigation scope.
Former next-step references are superseded by the current issue and stage ledger.

Stage 2 verification: APISIX profile, secret-safe error reporting, two TLS flag
values, effective deadline/retry policy, cached registration, ordinary client
tool loop, multipart text/tool streams, single-delta rejection, malformed API
responses and archive/macro reuse checks pass. The affected plugin, core LLM,
Open WebUI, recovery and chat regression checks pass with existing expected-failure
and XPASS outcomes retained. Ruff, Black, mypy, Cog documentation checks and
sdist/wheel packaging pass. These checks use synthetic protocol fixtures.

First workplace artifact (issue comment `6124613360`): the checkout matches
`d361179` and editable installation succeeded in `Projects\.venv`. Subsequent
bare `python` commands failed on import of `sqlite_utils`, which the installation
log confirms exists in that virtual environment. The profile/key/discovery commands
never reached execution. Workplace instructions now select one explicit environment
executable for every command. No production change is required for this mismatch.
The following workplace artifact resolves those import and discovery checks.

Second workplace artifact (issue comment `6124690308`, checkout `2020b30`):
the explicit virtual-environment executable successfully imports this checkout,
creates the APISIX profile with the selected TLS flag, stores the key, and lists
`deepseek-v4-flash`, `deepseek-v41-flash` and `glm-5.3` through authenticated
discovery. This verifies the real profile/authentication/discovery path, not model
inference. The KB command again used bare `python` and failed before assessment.
The directory listing confirms local `SKILL.md`, `SPEC.md` and a bundle directory
exist; completeness and macro resolution remain unverified. No KB recreation or
production-code fix is justified by this artifact. The next workplace commands
use the same explicit executable for KB assessment and the first model report.

Third workplace artifact (issue comment `6124772289`, implementation `2020b30`,
Python 3.12.1): all five API probes pass for `deepseek-v4-flash`, including
incremental text, streamed function arguments and synthetic tool-result
continuation. The text stream has six text deltas, first text at approximately
0.953 seconds and completion at 1.078 seconds. This verifies short synthetic
requests, not the long investigation or recovery workflow. The other two models
have discovery evidence only. Provider reasoning defaults are unchanged.

The same artifact assesses the existing local TIDE folder as structurally
`reusable`, with 455 rule triples, 360 macro files and no reported problems.
Preserve and reuse this bundle; no recreation or upload is indicated. This
assessment does not establish model-side KB context or semantic correctness of
every rule. The APISIX provider currently assesses the folder but does not yet
inject it into model prompts; that integration remains Stage 4 work.

The user's project lead describes APISIX as a proxy. APISIX routes traffic to
configured upstreams; this statement does not establish that the workplace model
route exposes existing Open WebUI knowledge or MCP APIs. Preserve the old remote
KB as well. Read-only Open WebUI knowledge/tool listings and confirmation of
the approved direct MCP transport/authentication method are the next workplace
evidence, without requesting URLs or credentials here.

External blocker B1 (Stage 3, current R4/R5/E1-E4): the development host cannot
reach corporate services. The latest user report establishes that the synthetic
task finished without crashing and showed no time awareness; it supplies no
measured duration or raw reasoning evidence. Completed-run timing plus workplace
multi-turn, reopen and cancellation/retry artifacts resolve the outstanding
criteria. Stage 4's final observed endurance envelope cannot be completed without
those artifacts. No MCP configuration is required to resolve this blocker.
