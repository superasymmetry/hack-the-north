"""Reading inventory and game statistics out of a MinecraftSim `info` dict.

Both controllers express their stop conditions in these terms ("three logs", "one sheep
killed"), so this is deliberately free of any model dependency.
"""
from collections import Counter
from typing import Any, Dict

import numpy as np

#: Slot types the game uses for "nothing here", which must not be counted as items.
EMPTY_ITEMS = frozenset({"none", "air", "minecraft:air", ""})


def bare_name(name: str) -> str:
    """Strip the namespace: "minecraft:oak_log" -> "oak_log"."""
    return name.lower().split(":")[-1]


def count_item(counts: Counter, item: str) -> int:
    """How many of `item` a Counter holds, tolerating the name an LLM is likely to say.

    Exact match wins; failing that every key containing `item` is summed, so "log" answers
    with oak_log + birch_log + ... and a plan does not have to spell items the way the
    1.16.5 item registry does.
    """
    item = bare_name(item)
    exact = sum(value for key, value in counts.items() if bare_name(key) == item)
    if exact:
        return exact
    return sum(value for key, value in counts.items() if item in key.lower())


def inventory_counts(info: Dict[str, Any]) -> Counter:
    """Total quantity per item type across the 36 inventory slots.

    `info['inventory']` is {slot_id: {'type': str, 'quantity': int}}, one entry per slot, so
    the same item in two stacks appears twice and has to be summed.
    """
    counts: Counter = Counter()
    for slot in (info.get("inventory") or {}).values():
        item_type = str(slot.get("type", "none"))
        if item_type.lower() in EMPTY_ITEMS:
            continue
        counts[item_type] += int(slot.get("quantity", 0))
    return counts


def stat_counts(info: Dict[str, Any], stat: str) -> Counter:
    """Cumulative game statistics, e.g. stat='mine_block' or 'kill_entity'.

    HumanSurvival observes these through ObserveFromFullStats, so `info['mine_block']` maps
    Minecraft's own keys ("minecraft.mine_block:minecraft.oak_log") to running totals. The
    exact spelling varies by item, hence the substring matching in the stop conditions.
    """
    counts: Counter = Counter()
    for key, value in (info.get(stat) or {}).items():
        counts[str(key)] += float(np.asarray(value).reshape(()))
    return counts
