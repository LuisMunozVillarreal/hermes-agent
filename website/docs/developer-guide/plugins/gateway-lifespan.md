# Required gateway plugin resources

Register `ctx.register_hook("gateway_lifespan", factory)` where
`factory(*, gateway)` returns an asynchronous context manager. This is an owned
resource contract, not an observer hook. It does not use fail-open hook dispatch
and is unavailable to shell hooks.

The gateway discovers launch-profile plugins, enters resources in registration
order before startup work or adapter connections, and exits them in reverse order
after active-work drain and before recording a clean exit. Entry failure prevents
startup and unwinds resources already entered. Aborted startup also unwinds them.
Stop during entry cancels and awaits the startup owner. A duplicate start is refused
without closing the existing owner's resources. Concurrent stops await the same exit
task; cancellation is re-raised only after exit finishes. New starts remain refused
until all stop callers have finished, so a retiring stop cannot close a successor.
The resource remains owned until then, including during failed-start unwind.
The existing gateway shutdown
watchdog bounds stuck exits at process level; it does not mark them clean.

Resources are a launch-profile boot snapshot. Plugin reload/unload changes future
registrations, not entered resources: those stay owned until gateway stop. Restart
the gateway to apply resource settings or remove a running resource. Discovery in a
secondary profile does not enter resources or create another owner.

Factories run on the gateway loop and must not block it. Move blocking work to an
owned context-propagating thread; await readiness before yielding. Handle partial
entry failure and cancellation inside the context manager. Exit must stop and join
owned work, not abandon a thread when cancellation arrives. Plugins retain their own
cross-process ownership fences and persistent recovery semantics. Registering a
factory alone must not start services, including in CLI and isolated worker clients.

The concrete consumer is the out-of-tree worker-queue plugin, which keeps its
coordinator inside the gateway process while systemd isolates worker execution.
No queue-specific configuration or implementation lives in core.
