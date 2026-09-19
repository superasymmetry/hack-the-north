"""Hold one Minecraft instance open so successive runs can skip the JVM boot.

Roughly a third of the "Resetting environment..." wait is Minecraft starting up: in
`logs/mc_*.log` it is consistently 11-14s from the JVM's first line to
`***** Start MalmoEnvServer`, before any mission is even sent. `MinecraftSim` pays that every
run, because it launches its own Minecraft inside `reset()` and kills it again in `close()`.

MineRL can instead attach to a Minecraft that is already running -- that is what
`MinecraftInstance(existing=True)` is for -- and it deliberately never kills an instance it
did not launch (`_destruct` is guarded by `not self.existing`, malmo.py:652). So a long-lived
holder process can own one Minecraft and lend it to any number of successive runs:

    bash scripts/mc_server.sh              # terminal 1, leave it running
    MC_PORT=9000 bash scripts/rocket2.sh   # terminal 2, as many times as you like

Not covered by this: the ~19s the mission spends generating the overworld. That is a
separate (larger) win -- see `mcagents.minecraft.world`.
"""
import os
import socket
import tempfile
import time

DEFAULT_PORT = 9000

# minestudio is imported inside the functions: it drags in PyAV, and anything that wants an
# OpenCV window has to get its first one open before that happens (see mcagents.gui).


def is_listening(port: int) -> bool:
    """True if something is already accepting connections on the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def serve(port: int = DEFAULT_PORT) -> None:
    """Launch a Minecraft instance on `port` and keep it alive until Ctrl-C."""
    from minestudio.simulator.entry import check_engine
    from minestudio.simulator.minerl.env.malmo import InstanceManager

    check_engine(skip_confirmation=True)

    if is_listening(port):
        raise SystemExit(
            f"Port {port} is already in use -- a server may already be running. "
            f"Attach to it with MC_PORT={port}, or pick another port."
        )

    # InstanceManager picks its port from the pid ((17 * os.getpid()) % 3989, malmo.py:318),
    # which is fine when one process both launches and uses the instance and useless when
    # another process has to find it. Pin it so the port is something we can publish.
    InstanceManager._get_valid_port = classmethod(lambda cls: port)

    instance = InstanceManager.get_instance(os.getpid())
    instance.launch(replaceable=False)   # nothing here relaunches it, so not replaceable

    print(f"\nMinecraft ready on port {instance.port}.")
    print(f"Run an agent against it with:  MC_PORT={instance.port} bash scripts/rocket2.sh")
    print("Ctrl-C here to shut it down.\n", flush=True)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down Minecraft.")
    # The atexit hook installed by launch() kills the process on the way out.


def attach(port: int = DEFAULT_PORT):
    """Register the Minecraft already running on `port` for the next MinecraftSim.

    `MinecraftSim.reset()` asks `InstanceManager.get_instance()` for an instance, and that
    hands back the first unlocked entry in `_instance_pool` before it considers launching
    anything (malmo.py:191). Dropping an `existing=True` instance in the pool first is
    therefore enough to make the whole reset path reuse it.
    """
    from minestudio.simulator.minerl.env.malmo import InstanceManager, MinecraftInstance

    if not is_listening(port):
        raise SystemExit(
            f"Nothing is listening on port {port} -- start a holder first with "
            f"`bash scripts/mc_server.sh`, or unset MC_PORT to launch Minecraft in-process."
        )

    # MineStudio's own InstanceManager.add_existing_instance() is the natural entry point,
    # but as shipped it calls MinecraftInstance(port=..., existing=True) without the required
    # `working_dir` positional (malmo.py:255 vs :350) and raises TypeError.
    #
    # The working_dir must also be one of ours rather than the server's: __init__ registers it
    # with MineStudio's garbage collector, which rmtree's it once this process exits
    # (database_manager.py:80). Pointing it at the server's directory would delete the running
    # instance's runtime out from under it.
    instance = MinecraftInstance(tempfile.mkdtemp(prefix="mc-attach-"), port=port, existing=True)
    InstanceManager._instance_pool.append(instance)
    InstanceManager.ninstances += 1
    return instance
