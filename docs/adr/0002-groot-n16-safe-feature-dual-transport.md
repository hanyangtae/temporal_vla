# Support dual SAFE feature transports for GR00T N1.6

Status: Accepted

## Context

ADR-0001 chose a dedicated ZMQ feature server for initial GR00T N1.6 SAFE
wiring. That was the right compatibility path while reproducing upstream
GR00T RoboCasa behavior and collecting the first SAFE artifacts.

The codebase now also supports the project FastAPI evaluation interface for
GR00T N1.6. The HTTP server exposes `/act` and `/act_with_features`, and both
HTTP and ZMQ feature transports accept optional per-call `inference_seed`.
Both transports share the DiT pre-velocity feature capture implementation
through `src.policies.groot`.

## Decision

Maintain both transports:

- ZMQ `get_action_with_features` remains the upstream-compatible SAFE
  collection path and the reference for reproducing existing SAFE rollout
  artifacts.
- HTTP `/act` remains the project-wide VLA evaluation interface for GR00T.
- HTTP `/act_with_features` is a supported SAFE feature collection transport
  when the caller needs the project FastAPI interface.

The feature definition must stay transport-independent. ZMQ and HTTP must call
the same feature-capture module and emit metadata that normalizes to the same
SAFE pkl schema:

- feature kind
- feature axes
- feature slice
- exported action-token count
- feature, valid, and model action horizons
- denoising-step count
- optional call-local inference seed semantics

GR00T RoboCasa observation/action key conversion and scenario replay live under
`src.policies.groot`. SAFE feature metadata normalization is policy/version
agnostic and lives in `src.policies.safe_metadata`. Script entrypoints may adapt
CLI, process, or transport details, but should not own those shared contracts.

## Consequences

The old "ZMQ only" boundary is no longer accurate for new work. The boundary is
now:

- `src.policies.groot`: shared GR00T RoboCasa policy contracts and runtime
  behavior.
- `scripts/safe/groot_n16/robocasa/serve/feature_server.py`: ZMQ adapter.
- `scripts/serve/groot.py`: FastAPI adapter.
- `scripts/safe/groot_n16/robocasa/collect`: rollout orchestration and SAFE
  pkl writing.

This keeps existing ZMQ artifacts reproducible while allowing HTTP-based SAFE
feature collection to use the same feature semantics.

Future refactors should avoid moving GR00T-specific SAFE semantics back into a
transport entrypoint. If ZMQ and HTTP diverge, the shared module or tests should
be fixed first.
