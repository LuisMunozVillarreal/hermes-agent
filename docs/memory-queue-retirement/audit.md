# Postrestart audit, main agent only

Window: 2026-10-27 12:32:54 to 13:29:12 UTC. Canonical state.db inventory: 5 Discord threads, 12 sessions including children, 609 message records. Journal independently captured for the same interval. This is Hermes-recorded activity, not a server-wide Discord history census. No workers spawned; no runtime changes.

## Findings

- Three initial delegations omitted retained memory: recording extension 1024 MiB peak; home cleanup 1600 MiB; /agents installation 1200 MiB. Each received growth_required. Mandatory-retained/headroom and no-retry fix 843296ee is published but absent from live checkout. Replacement jobs explicitly reserved child capacity.
- Recording replacement diagnostic b072b25b441a4280b86ba9ea07983970 remains unexplained. Two services began stopping at 13:13:00 before EOF. No initiating receipt links this to cancellation, owner loss, growth denial or recovery. Do not attribute its stop to OOM.
- Home-cleanup terminal b4345de4683847448fd5d96a890d2475 has direct systemd OOM evidence at 13:22:31. Command estimate 128 MiB, journal peak 160M. Public result preserves unknown exit status but coordinator state failed/reason null loses OOM attribution. This is a separate incident, not proof of whole-host exhaustion or the recording stop cause.
- Node tests and uv environment build show thread creation EAGAIN/SIGABRT. enforcement/linux.py hardcodes TasksMax=64. This is a concrete likely constraint, not historically proven without pids.events. /agents worker reports bounded concurrency/CPU affinity recovered its build. Capture PID counters before teardown and test a configurable measured task budget rather than blindly removing protection.
- Cleanup search_files returned generic execution-owner launch failed. Same-interval journal at 13:17:44 records hwq-0638f79e49fa4c68ae9c02ef2ffe533d CHDIR failure. Preserve safe launch stage and errno/unit ID through workload_boundary.py; confirm correlation with persistent ticket receipts.
- File tools ask for 1 GiB child capacity in observed refusals. Audit default estimate and expose it to parent budgeting; do not equate nonzero slack with enough slack.
- /agents worker could not push using OpenClawFeexBot. Local live checkout now 57c51c8a03 above cd574cb65c; gateway still PID3301714 started12:32:54. Mandatory-memory fix is not installed. Reconcile local/remote changes before deployment, no reset.
- Watch thread contains two messages and no execution failure. Ordinary command mistakes, missing paths, SSH quoting, missing python/ruff and intentional red tests are excluded from scheduler-defect totals. Recording app owner reports readiness fixed; do not repeat its implementation here.

## Required logging points

1. coordinator.py cleanup: write durable stop-request receipt BEFORE recursive child stop/backend.stop. Include cause enum, initiating ticket/parent ticket, requesting owner PID+start identity, session/delegation correlation, unit/invocation, requested outcome, state, wall+monotonic timestamp. Propagate original cause through child teardown.
2. coordinator.py cancel, recover, _tick, monitor_growth: distinguish user cancel, caller disconnect, owner gone, quarantine retry, restart recovery, observed process completion, growth denied, growth step exhaustion, drain. Log growth decision with current usage/cap, requested cap, count and available headroom. Existing generic failed/cancelled is insufficient.
3. enforcement/linux.py start/completed/stop/grow: capture systemd Result, ExecMainCode/Status, ActiveState/SubState, InvocationID and cgroup memory.events/pids.events/current/max before stopping or unloading. Log stop request, command result and escalation separately. completed currently collapses OOM and other failures to failed.
4. agent/workload_boundary.py launch catch: safe diagnostic ID, stage, exception class/errno and ticket rather than raw exception/command/environment.
5. ledger.py finish/prune: retain small payload-free terminal receipts independently of short-lived admission rows; read them via disposition after pruning. Bound size/retention; no prompts, credentials or commands.
6. isolated_agent.py and delegate_tool_progress.py: preserve coordinator cause/provenance in failure output, do not present synthetic255 as known child exit; remove unconditional retry-with-smaller-task advice.

Priority: activate existing validated memory fix after reconciling branch drift; durable lifecycle receipts and native OOM/PID attribution; then targeted PID/launch fixes supported by deterministic tests. Logging improves attribution and does not itself fix the unknown stop.
