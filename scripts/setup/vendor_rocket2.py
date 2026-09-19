"""Fetches the two ROCKET-2 source files this project imports, and patches them for timm 1.0.

Why this exists: the ROCKET-2 policy class is *not* in the `minestudio` pip package.
MineStudio 1.1.6's model gallery does not include it -- ROCKET-2 lives in the
CraftJarvis/ROCKET-2 repo, and only on its `master` branch (the `main` branch
README still says "code released soon"). Only two of its files are actually needed to run
the policy: `model.py` (the `CrossViewRocket` class, a `MinePolicy` subclass, so it plugs
into MineStudio's usual `get_action()` loop) and `cfg_wrapper.py` (classifier-free guidance
over two forward passes).

Rather than committing a silent fork, this script downloads both into
mcagents/vendor/rocket2/ and applies the one edit they need here. Re-run it to pick up
upstream changes; it overwrites whatever is there, so don't hand-edit the vendored files.

Usage:
    conda activate ./.conda-env
    python scripts/setup/vendor_rocket2.py
"""
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/CraftJarvis/ROCKET-2/master/{}"
VENDOR_DIR = Path(__file__).resolve().parents[2] / "mcagents" / "vendor" / "rocket2"
FILES = ["model.py", "cfg_wrapper.py"]

# timm >= 1.0.13 refuses a bare 'timm/vit_base_patch16_224.dino' -- anything that looks like
# a Hub id now needs an explicit source prefix ("Use 'hf-hub:...' to load from the Hub", from
# timm/models/_factory.py:parse_model_name). ROCKET-2 was written against an older timm and
# hardcodes the bare names as default arguments, which are also what the published
# checkpoint's config.json carries, so this cannot be fixed by passing a different argument
# from our side -- it has to be normalized inside the constructor.
TIMM_SHIM = '''

def _timm_name(name: str) -> str:
    """Prefix a bare Hub id for timm >= 1.0.13, which requires an explicit model source."""
    if "/" in name and ":" not in name:
        return f"hf-hub:{name}"
    return name

'''

PATCHES = {
    "model.py": [
        {
            "name": "insert the _timm_name() shim after the imports",
            "old": "from minestudio.utils.register import Registers\n\nBINARY_KEYS",
            "new": "from minestudio.utils.register import Registers\n" + TIMM_SHIM + "\nBINARY_KEYS",
        },
        {
            "name": "view_backbone: bare timm Hub id -> hf-hub: prefixed",
            "old": "timm.create_model(view_backbone,",
            "new": "timm.create_model(_timm_name(view_backbone),",
        },
        {
            "name": "mask_backbone: bare timm Hub id -> hf-hub: prefixed",
            "old": "timm.create_model(mask_backbone,",
            "new": "timm.create_model(_timm_name(mask_backbone),",
        },
    ],
}

INIT_PY = '''"""Vendored ROCKET-2 sources -- do not hand-edit, see scripts/setup/vendor_rocket2.py."""
from .model import CrossViewRocket, load_cross_view_rocket
from .cfg_wrapper import CFGWrapper

__all__ = ["CrossViewRocket", "load_cross_view_rocket", "CFGWrapper"]
'''


def main() -> int:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        url = RAW.format(filename)
        print(f"fetching {url}")
        source = urllib.request.urlopen(url).read().decode()

        for patch in PATCHES.get(filename, []):
            if patch["old"] not in source:
                print(f"  [!] {patch['name']}: anchor not found -- upstream changed, patch by hand")
                return 1
            source = source.replace(patch["old"], patch["new"])
            print(f"  patched: {patch['name']}")

        (VENDOR_DIR / filename).write_text(source)

    (VENDOR_DIR / "__init__.py").write_text(INIT_PY)
    print(f"wrote {len(FILES) + 1} files into {VENDOR_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
