# APISIX profile and availability specification

Authoritative job and artifact coordination: [XSoar-MDR #28](https://github.com/Darkaxt/XSoar-MDR/issues/28).
Authorization: already authorized by the user's request to merge Open WebUI,
focus on APISIX, create the first client profile, assess KB reuse, and run
availability tests. Corporate infrastructure is inaccessible here; credentials
remain on the work computer.

The issue's R1-R7 requirements remain authoritative. This document maps the
first implementation to those requirements without claiming migration completion.

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
- A3 (R4): Assess the existing local TIDE v0.3 folder or archive using the
  existing virtual-folder reader. Verify instructions, complete rule triples,
  JSON readability and macro references. Report reusable/incomplete/not
  configured without uploading, deleting, rebuilding or altering knowledge.
  Local KB content injection and real MCP integration belong to stage 4.
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

| Stage | Status | Evidence / remaining criteria |
|---|---|---|
| 1. Preserve Open WebUI baseline | COMPLETE | PR #1 merged as `31b9bce`; regression/dependency, lint, formatting, type, documentation and packaging QA inspected |
| 2. Profile and availability diagnostics | COMPLETE | A1-A4 verified with protocol/client-loop fixtures and affected regressions; A5 setup/report instructions delivered in issue #28 |
| 3. Workplace availability validation | BLOCKED | Profile/discovery, selected-model inference/stream/tool continuation and local KB structure verified; independent Splunk MCP access remains unverified |
| 4. Investigation integration and validation | NOT STARTED | R4-R7 depend on confirmed workplace model/MCP contracts |

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

External blocker B1 (Stage 3, R3-R5): the development host cannot reach corporate
services, has no corporate credentials or actual TIDE folder, and has no direct
MCP transport/authentication contract. Workplace artifacts now verify discovery,
short model/stream/tool behavior and local knowledge structure. R5 remains blocked
on approved independent MCP access, transport and authentication metadata from
the workplace. That evidence resolves the remaining Stage 3 blocker; existing
Open WebUI tool visibility alone cannot satisfy independent MCP availability.
Stage 4 KB-context/MCP integration and the final investigation/recovery workflow
cannot be considered complete. No required work is deferred outside these named
stages.
