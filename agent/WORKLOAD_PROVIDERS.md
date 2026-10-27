# Workload provider API v1 (experimental)

A profile-scoped provider is registered with
`PluginContext.register_workload_provider(provider)`. Its `name` must be unique;
`api_version` must match `WORKLOAD_API_VERSION`. Unload restores displaced
registrations through the existing plugin lifecycle. Provider selection is keyed
by `hermes_home_key`, preserving multiplex profile isolation.

`workloads.mode` is `disabled` by default. `required` selects the explicitly named
`workloads.provider`; missing/incompatible registration, bad config and failed
execution calls raise `WorkloadUnavailable`. Required never falls back to Popen.
Providers advertise an immutable `supported_kinds` set. Wired kinds are:

| Kind | Core execution path |
| --- | --- |
| `terminal` | Local foreground, background and timeout-to-background launches |
| `terminal-pty` | Managed local background PTY launch via the same workload boundary |
| `cron-script` | No-agent scripts, agent pre-run scripts, inline prompt scripts and monitor scripts |
| `delegation` | Isolated child AIAgent, including live single-goal interception and per-task batches |
| `cron-agent` | Isolated scheduled AIAgent; estimate captured before the watchdog thread starts |

Remote terminal adapters and external ACP delegation remain unsupported in
required mode. Ordinary gateway turns, browser processes and in-process monitor
URL fetching are not managed execution groups.

## Execution ownership

`popen(kind, argv, config=..., **kwargs)` is an execution-owner operation, not an
advisory hook. It must reserve before spawn, return an appropriate Popen/PTY handle,
retain ownership after background/yield, and verify descendant cleanup before
release. Managed handles set `_hermes_managed = True` so termination goes through
the execution owner instead of PID-only upstream cleanup. Disabled paths retain
upstream process/scope handling. A metadata `submit(WorkloadSpec)` ticket alone is
never authorization for core to launch a workload.

Required-mode delegation and agent cron use `agent/isolated_agent.py` and
`agent/isolated_worker.py`, not an in-process fallback. Their versioned JSON-only
control protocol carries allowlisted construction/run data and supports results,
approval relay, steering and cancellation. Live agents, clients and callbacks are
not serialized. These adapters and their fidelity tests are experimental; their
existence does **not** establish full safety review or rollout readiness.

## Native memory estimates

Model-facing fields are static schema properties, not live telemetry. At tool
definition construction, required memory-managed providers make peak mandatory
for terminal calls and each delegation task. Cached conversation schemas are not
mutated. Runtime validates the same requirement even with operator defaults:

- `terminal.estimated_peak_memory_mb`: this execution's complete process-tree peak.
  `terminal.estimated_retained_memory_mb` is optional, for the execution unit's own cap.
- `delegate_task.tasks[i].estimated_peak_memory_mb` and
  `estimated_retained_memory_mb`: independent per-child estimates. The legacy
  single-`goal` Python/dispatch shape also forwards these top-level fields; the
  public model schema continues to advertise the `tasks` shape.
- Cron create/update use the same two fields, persisted on the job and returned by
  list/update responses. The CLI accepts `--estimated-peak-memory-mb` and
  `--estimated-retained-memory-mb` on `hermes cron create` and `hermes cron edit`.

Despite the historical `_mb` spelling, values are **integer MiB**, not decimal MB.
`agent/workload_estimates.py` validates without coercion and multiplies by
`1024**2` at the input boundary. Booleans, strings, floats (including integral
floats), zero, negatives and values that overflow signed 64-bit bytes are rejected.
An omitted value stays absent. Retained memory requires a peak and cannot exceed
it: parent memory is **included in**, not added to, the tree peak. Unmanaged and
legacy non-memory-managed providers still accept omitted estimates. Automatic cron
fires and internal non-model terminal calls retain their configured-default path.

With the queue provider, an explicit peak and omitted retained cap resolves to
retained=peak, ignoring configured retained defaults. This leaves **zero child
slack**. A worker that will launch separately admitted commands should supply a
realistic retained cap and a peak large enough for that cap plus concurrent child
peaks. Ordinary shell descendants within one execution unit also consume its cap.
For example, a 2048 MiB parent cap plus a 512 MiB nested command needs at least
2560 MiB subtree peak. These are example budgets, not a minimum or a guarantee.

The authenticated `agent.workload_boundary.request_growth` API requests an
absolute target. Growth takes effect only after scheduler approval; increasing
peak alone preserves the own-unit cap and adds child slack. Neither model terminal
dispatch nor delegation automatically calls this API for a queued child. A child
that cannot fit its parent's current envelope can stay queued until cancellation,
an explicit queue TTL, or approved parent growth. Telemetry-based own-unit growth
is not a remedy for a blocked child. Do not retry with ever larger child estimates.

`WorkloadValidationError` crosses the execution boundary with fixed reason text
and validated integer byte counts only. Arbitrary legacy provider exceptions keep
the generic launch error; config, payloads and credentials are never interpolated.

The provider receives `peak_memory_bytes` and optionally `retained_memory_bytes`
as separate launch keyword arguments. It must consume them rather than forwarding
them to `subprocess.Popen`. Request-local ContextVars avoid shell-environment
mutation and concurrent-job leakage. Isolated agents capture bytes at construction
and restore them at launch in the worker thread. Each cron script entry point
resets the scope from its own job, including legacy jobs with no estimate, and
restores the previous scope on exit. Cron scripts and the agent are separate
launches using the same job estimate, not one reservation made at job creation.

Creating or editing a cron job reserves nothing. Each execution is admitted by
the provider. In required mode, the consumer must reject missing estimates unless
an operator explicitly configured its default policy. Existing jobs are not
silently assigned estimates or migrated. For example, after reviewing one job:

```sh
hermes cron edit JOB_ID --estimated-peak-memory-mb 768 --estimated-retained-memory-mb 400
```

This changes only that job's estimates; it neither enables a paused job nor runs
it. The core does not invent a default or import provider-specific scheduling
policy. With protection disabled, valid estimate metadata does not select a
provider or change upstream process ownership.

## Authority and scope

Estimates are not leases. Model schemas expose neither lease/ticket/generation
fields nor byte-level ownership metadata. `workload_config()` removes configured
`parent_lease` values and obtains the parent only from the authenticated worker
context. Model-created nested terminal/delegation work must supply its own peak
under memory-managed admission, never inherit an unrelated launch's estimate.

No queue policy, cgroup backend or provider module name belongs in core. The
consumer implementation and its acceptance/operations records live separately in
LM3Labs/hermes/apps/hermes-worker-queue. Kernel enforcement and descendant cleanup
must be verified against that real consumer; fixture-provider/unit passes alone
are not evidence that enforced mode is ready. No runtime activation is implied.
