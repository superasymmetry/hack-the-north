"""Dry-run the city layout without launching Minecraft.

    python -m mcagents.cli.city            # the whole command list
    python -m mcagents.cli.city --summary  # just the shape of it

Iterating on the layout this way costs nothing; iterating in-game costs a 30s boot and 20s
of chat commands per look.
"""
import argparse

from mcagents.minecraft.city import CityBuilder, CityLayout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ground", type=int, default=70, help="street level y")
    parser.add_argument("--summary", action="store_true", help="do not print the commands")
    args = parser.parse_args()

    layout = CityLayout()
    builder = CityBuilder(0, 0, args.ground, layout=layout, seed=args.seed)
    commands = builder.commands()

    print(f"{len(commands)} commands, ~{len(commands) / 20:.0f}s of game ticks")
    print(f"plot {layout.span}x{layout.span} from {layout.low} to {layout.low + layout.span - 1}, "
          f"lots at {[layout.lot_bounds(i) for i in range(layout.lots)]}")
    if not args.summary:
        for command in commands:
            print(command)


if __name__ == "__main__":
    main()
