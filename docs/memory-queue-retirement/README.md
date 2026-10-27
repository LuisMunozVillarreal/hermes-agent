# Memory queue retirement and resumption handoff

Retired at the user's request on 2026-10-27. Do not resume this experiment without explicit authorization. Work on retirement was performed in the main agent, without delegated workers.

## Preserved source

- Core archive: archive/memory-queue-1564636327952261141. Includes published core 843296ee0af10ec2438526c78532861e9d642304 plus the local /agents links change originally 57c51c8a032dd8c84f98009bfd33c4e14746ad11.
- Separate exact local backup: archive/memory-queue-local-1564636327952261141.
- Plugin: LM3Labs/hermes, apps/hermes-worker-queue, merge 4ad73fab336ac1b756e39d1b02fc63bfeb219f45, PR24. Plugin history stays in that repository; no deletion of its source is required for deactivation.
- Installed pair before retirement: core cd574cb65c7188a7699fe0627b515d1a855ef1ef plus /agents local change, plugin PR24. Core 843296ee was published but never activated.
- Rebuilt default-execution core: af77acb749d1860166b8d9f92ef898f97610fec9 plus the preserved /agents commit. The queue merge and its feature-only ancestors/descendants are omitted, not reverted. See removed-commits.txt.
- Private local evidence and verified pre-removal bundle: ~/.hermes/cache/scratch/postrestart-audit-main/. Raw evidence can contain private conversation content; do not commit it to this repository.

## Standard behavior selected

The selected Hermes base defaults to delegation.max_concurrent_children=10. User requests 30. The standard implementation has no separate configurable 100-waiting-job admission queue; its background-capacity fallback is synchronous execution. This retirement must not introduce a replacement scheduler or pretend max_concurrent_children is a global process ceiling. The setting also limits submitted batches and background delegation as implemented upstream. Ordinary upstream terminal protection remains; the custom memory estimates, queue and isolated-agent execution are removed.

## TODO if explicitly resumed

1. Reassess whether per-command/model memory estimates are worth the operational complexity. Do not assume models can reliably predict peaks. Measure wrapper overhead and representative tools first. Preserve user decision to prefer standard execution unless changed explicitly.
2. Merge/rebase the archive into a disposable branch against the then-current integration, never reset integration to this stale archive. Preserve unrelated later changes. Revalidate plugin/core contract compatibility.
3. Validate mandatory model delegation peak+retained, realistic child headroom and immediate non-retry validation behavior in 843296ee. Its focused gate passed 102 tests but it was never activated. Nonzero slack alone does not satisfy default file-tool estimates of 1 GiB.
4. Investigate unknown recording-worker stop diagnostic b072b25b441a4280b86ba9ea07983970, PID3307939, at13:13:00 UTC. Explicit-headroom replacement stopped before EOF; initiating cause is unknown. Do not call it OOM based on a separate incident.
5. Add payload-free durable lifecycle receipts BEFORE cleanup. See audit.md for exact coordinator/enforcement/ledger/core logging points. Preserve original stop cause through recursive child cleanup, requester identity, owner liveness, growth decision, systemd invocation, OS exit/signal status, memory/PID counters and escalation. Retain bounded receipts independently of queue-row pruning.
6. Preserve native OOM attribution. Home-cleanup terminal b4345de4683847448fd5d96a890d2475 at13:22:31 has explicit systemd oom-kill evidence; tool output only exposed unknown EOF and failed/reason=null. Command estimated128 MiB; journal peak160M. Do not infer whole-host exhaustion.
7. Investigate TasksMax=64 against Node/uv thread-creation EAGAIN and SIGABRT. Capture pids.events/current/max before teardown. Reproduce in isolated staging and make limits explicit/measured if retained. Do not silently remove host protection.
8. Preserve safe launch-stage failures in agent/workload_boundary.py. Home-cleanup generic launch failure coincided with systemd CHDIR failure. Use ticket/diagnostic correlation and allowlisted errno/stage, never raw environment or command disclosure.
9. Remove unconditional smaller-task retry advice and preserve unknown completion provenance through child accounting and notification formatting.
10. Re-run pure regression suites and native staging with no production coordinator interference. Previous combined acceptance reported457 tests/15 runtime cases but did not prove these later failure paths. Test owner loss, cancel, growth denial/exhaustion, telemetry failure, OOM, PID exhaustion, CHDIR failure, transport EOF and pruning.
11. Activation must be an explicitly approved future operation, core/plugin compatibility verified first. Back up current config, restore only deliberate settings, inspect manager ownership and active jobs, and have the user restart. An archived commit or installed file is not proof of activation.

## Configuration to reconstruct, not blindly restore

Historical provider workloads.mode=required, provider=hermes-worker-queue, state=~/.hermes/worker-queue, backend=local, queue_ttl_seconds=0 and runtime_seconds=0. Delegation running cap0 meant memory-managed unlimited. Default delegation/cron-agent peak4GiB, retained2GiB; terminal/file-related defaults require inspection of exact archived code. Plugin gateway enabled with max_waiting100, max_running0, per_owner0, max_records64, retention_seconds3600, runtime_seconds0/max_runtime_seconds0. These are historical values, not recommended settings.

Keep runtime state and configuration backups private. Restore neither old credentials nor unrelated profile/configuration values from a historical backup. Retirement itself does not authorize stopping unrelated jobs or restarting the gateway.
