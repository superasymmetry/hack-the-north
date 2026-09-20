"""Load a pre-generated world instead of regenerating it on every reset.

`MinecraftSim(seed=0)` is fixed and `HumanSurvival` asks for
`<DefaultWorldGenerator forceReset="true"/>`, so every run builds the same world from
scratch -- every run in `logs/mc_*.log` spawns at exactly (-3009.5, 71.0, -5572.5). That
costs ~19s of the reset: ~6s creating the level and ~13s under `Preparing start region`.

The engine can load a saved world instead. `HumanSurvival(load_filename=...)` emits
`<LoadWorldFile>` into the mission XML, and `EnvServer.loadOrCreateWorld` then calls
`ReplaySender.loadWorldFromZip(path)` in place of `createNewWorld` -- measured here,
22.7s -> 9.0s for the same reset against the same instance.

    bash scripts/build_world.sh      # generates a world the slow way, saves worlds/plains.zip
    bash scripts/rocket2.sh          # picks it up automatically

Caveats worth knowing:

- The zip *is* the seed. `MinecraftSim(seed=...)` and `preferred_spawn_biome` are both read
  inside `createNewWorld`, exactly the branch loading skips, so neither has any effect while
  a world is loaded. `WorldCache.build()` sets the biome itself, and the biome is part of the
  cache filename so a run cannot silently reuse a world generated for a different one.
- Each load unzips into a fresh `$TMPDIR/<random hex>` and leaves it there, so runs
  accumulate ~4MB apiece in /tmp.
- The player state saved into the world comes along with it. Capture right after a reset, as
  `build()` does, or you will be reloading whatever the agent had done by then.
"""
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Where the agent spawns. A property of the cached world rather than a per-run setting, for
#: the reason in the module docstring.
DEFAULT_BIOME = "plains"

WORLDS_DIR = Path(__file__).resolve().parents[2] / "worlds"

#: `mcagents.minecraft.bake` writes its output as `<terrain>_city.zip`, or plain `city.zip`
#: for the default terrain. Such a name is a *world* name and not a biome, which matters on
#: the one path that reads the biome -- generating, when the zip turns out to be missing.
BAKED_SUFFIX = "city"

#: Worlds that are already a city, by name, though not baked by us: imported maps.
CITY_WORLDS = frozenset({"chiangtung"})

#: The world a run loads when MC_WORLD does not name one: the plains cache written by
#: `scripts/build_world.sh`. Runs generate a world the slow way while it is missing.
DEFAULT_WORLD = WORLDS_DIR / "plains.zip"


def terrain_biome(name: str) -> str:
    """The biome a cached world's name implies, seeing through the baked-city marker."""
    if name == BAKED_SUFFIX:
        return DEFAULT_BIOME
    if name.endswith(f"_{BAKED_SUFFIX}"):
        return name[:-len(BAKED_SUFFIX) - 1]
    return name


def baked_city_path(biome: str = DEFAULT_BIOME, directory: Path = WORLDS_DIR) -> Path:
    """Where a city baked onto `biome` terrain lives."""
    stem = BAKED_SUFFIX if biome == DEFAULT_BIOME else f"{biome}_{BAKED_SUFFIX}"
    return directory / f"{stem}.zip"


@dataclass
class WorldCache:
    """A generated-once Minecraft world on disk, and the two things you do with it."""

    #: The biome to generate, on the one path that generates: the zip being absent.
    biome: str = DEFAULT_BIOME
    directory: Path = WORLDS_DIR
    #: The file stem, when it differs from the biome -- `city`, for a baked world.
    name: Optional[str] = None

    @property
    def path(self) -> Path:
        return self.directory / f"{self.name or self.biome}.zip"

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def is_baked_city(self) -> bool:
        """Whether this world already has a city in its terrain, by name."""
        stem = self.name or self.biome
        return stem in CITY_WORLDS or stem == BAKED_SUFFIX or stem.endswith(f"_{BAKED_SUFFIX}")

    def apply(self, sim) -> None:
        """Make `sim` load this world on every reset from here on.

        `EnvSpec.reset()` rebuilds the handler lists at the top of every `env.reset()`, and
        `HumanSurvival.create_agent_start` appends `LoadWorldAgentStart(self.load_filename)`
        whenever that attribute is set -- so setting it on the live task is enough, with no
        patching of the installed minestudio.
        """
        sim.env.task.load_filename = str(self.path.resolve())

    def build(self) -> Path:
        """Generate a world the slow way once and save it.

        Deliberately launches its own Minecraft rather than attaching to a holder: the world
        is written under the working directory of whichever process launched the instance,
        and an attached client does not know the holder's.
        """
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.entry import check_engine

        check_engine(skip_confirmation=True)
        print(f"Generating a {self.biome} world (this is the slow reset -- once)...", flush=True)
        sim = MinecraftSim(action_type="env", preferred_spawn_biome=self.biome)
        try:
            sim.reset()
            working_dir = Path(sim.env.instances[0].working_dir)
            saves = sorted((working_dir / "saves").glob("mcpworld*"), key=lambda p: p.stat().st_mtime)
            if not saves:
                raise SystemExit(f"No world was written under {working_dir / 'saves'}.")
            out = self.pack(saves[-1])
        finally:
            sim.close()

        print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB) from {saves[-1].name}.")
        print("Later runs will load it automatically.")
        return out

    def pack(self, save_dir: Path) -> Path:
        """Zip a Minecraft save directory into the layout `loadWorldFromZip` expects.

        That method takes the world name from `entries[0].split("/")[2]` and then loads
        `<extract_root>/saves/<name>` -- so entries have to be `./saves/<name>/...`, where the
        leading "." is doing the real work of making index 2 land on the world name.
        `zipfile.write()` normalizes "./" away, hence the hand-built ZipInfo.
        """
        save_dir = Path(save_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(save_dir.rglob("*")):
                if not path.is_file():
                    continue
                name = f"./saves/{save_dir.name}/{path.relative_to(save_dir).as_posix()}"
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
        return self.path

    @classmethod
    def from_path(cls, path: Optional[Path]) -> Optional["WorldCache"]:
        """A cache pointing at an explicit zip, e.g. from $MC_WORLD or --world."""
        if path is None:
            return None
        path = Path(path)
        return cls(biome=terrain_biome(path.stem), directory=path.parent, name=path.stem)
