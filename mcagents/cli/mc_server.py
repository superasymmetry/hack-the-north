"""Hold one Minecraft open so repeated runs can attach to it instead of booting a JVM each time.

    bash scripts/mc_server.sh               # leave running
    MC_PORT=9000 bash scripts/rocket2.sh    # in another terminal, as often as you like

Ctrl-C here shuts the Minecraft process down.
"""
import argparse
import os

from mcagents.minecraft.instance import DEFAULT_PORT, serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=int(os.environ.get("MC_PORT", DEFAULT_PORT)))
    serve(parser.parse_args().port)


if __name__ == "__main__":
    main()
