"""Run JarvisVLA against a live Minecraft from a plain English instruction.

    ./scripts/jarvisvla.sh                       # launches its own Minecraft
    MC_PORT=9000 ./scripts/jarvisvla.sh          # attaches to a scripts/mc_server.sh holder
    python -m mcagents.cli.jarvisvla --plan my_plan.json --max-steps 200

The policy is a 7B VLM and does not run on a laptop: start it on the GPU box with
scripts/serve_jarvisvla.sh, tunnel to it, and export JARVISVLA_URL. Sanity-check the
connection on one still frame before burning a run, and the decoder without either:

    python -m mcagents.cli.jarvisvla --frame logs/frame.png "Chop down the oak log."
    python tests/test_jarvisvla_actions.py

Each plan entry is an instruction plus a stop condition. JarvisVLA was post-trained on
instructions phrased like its own assets/instructions.json -- short, imperative, one
primitive each -- and that is the distribution to stay on; compositional goals degrade badly.

The run opens a window of what the model is looking at, captioned with the instruction and
the action it just chose; --no-preview (or JARVISVLA_PREVIEW=0) turns it off, and it turns
itself off when there is no display. An mp4 lands under logs/jarvisvla either way.

By default the game waits for the model, which at ~5 calls/s is a quarter of Minecraft's own
speed and looks it. --async runs the model on a worker thread instead: the game plays at 20
ticks/s and each decision is held until the next reply lands. Use it to demo, not to measure
-- the policy then acts on a frame that is one call old. See docs/jarvisvla.md.

The city is *not* built here by default: it exists to give a pointing agent something to
point at, and it costs ~20s of every reset. --city turns it on.
"""
import argparse
from typing import Optional, Sequence

from mcagents.agents.jarvisvla import (JarvisVLAAgent, JarvisVLAClient, JarvisVLAConfig,
                                       sim_kwargs)
from mcagents.agents.jarvisvla_actions import TOKEN_RE, decode_actions
from mcagents.cli.plan import Plan, load_plan
from mcagents.minecraft.session import EnvConfig, Session

#: "Hunt a pig." is verbatim from upstream's instructions.json under `kill_entity:pig`, which
#: is the distribution to stay on: the model maps an instruction to a *target*, so a phrasing
#: it has seen is worth more than a clearer one it has not.
DEFAULT_PLAN: Plan = [
    {"instruction": "Hunt a pig.",
     "stop": {"stat": "kill_entity", "match": "pig", "count": 1}},
]

#: A sword in slot 0, which is selected at spawn -- upstream's kill configs spawn a
#: diamond_sword and nothing else.
INIT_INVENTORY = [{"slot": 0, "type": "diamond_sword", "quantity": 1}]

#: Pigs, put where the agent is already looking. Upstream's kill tasks do not ask the policy
#: to *find* the mob: kill_zombie.yaml summons it into range_z [3, 10] straight ahead. That
#: matters more here than it looks -- the instruction barely conditions this checkpoint, so
#: what is in frame is most of what decides the action. A pig the agent has to go looking for
#: is a pig it will walk past.
MOBS = [{"name": "pig", "number": 3, "range_x": [-3, 3], "range_z": [3, 10]}]

# Upstream evaluates kill tasks at temperature 0.9, not the 0.6 its mining rollout uses
# (scripts/evaluate/rollout-kill.sh). JarvisVLAConfig's default follows the mining script, so
# for this plan: JARVISVLA_TEMPERATURE=0.9 ./scripts/jarvisvla.sh


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", help="JSON file of instructions; default is the one in this module")
    parser.add_argument("--mc-port", type=int, help="attach to a held instance on this port")
    parser.add_argument("--city", action="store_true", help="build the city (~20s a reset)")
    parser.add_argument("--max-steps", type=int, help="hard ceiling per instruction")
    parser.add_argument("--async", dest="async_predict", action="store_true", default=None,
                        help="think on a worker thread so the game runs at full speed "
                             "(smooth to watch; each decision is held for several ticks)")
    parser.add_argument("--fps", type=float,
                        help="ticks per second to hold the game to (default 20, Minecraft's "
                             "own rate; 0 removes the cap)")
    parser.add_argument("--log-every", type=int, default=25, help="how often to print the action")
    parser.add_argument("--no-record", dest="record", action="store_false",
                        help="do not write an mp4 of the run under logs/")
    parser.add_argument("--no-preview", dest="preview", action="store_false", default=None,
                        help="do not open a live window of what the model is looking at")
    parser.add_argument("--frame", metavar="IMAGE", nargs=2,
                        help="ask the model what it would do on one still frame, then exit")
    return parser.parse_args(argv)


def check_frame(path: str, instruction: str) -> None:
    """Ask the server what it would do on a single still frame.

    The cheapest way to tell "the server is wired up and answering in action tokens" from
    "the run is going nowhere".
    """
    import cv2

    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    config = JarvisVLAConfig.from_env()
    config.history_num = 0
    client = JarvisVLAClient(config)
    print(f"asking {config.urls[0]} ({client.model})")

    reply = client.complete(client.build_messages(instruction, frame, []))
    print(f"reply: {reply!r}")
    if not TOKEN_RE.search(reply):
        raise SystemExit("\n!! no action tokens in the reply. Either the server dropped "
                         "skip_special_tokens=False, or it is not serving JarvisVLA.")
    for index, action in enumerate(decode_actions(reply)):
        print(f"  action {index}: buttons={int(action['buttons'][0])} camera={int(action['camera'][0])}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.frame:
        return check_frame(*args.frame)

    plan = load_plan(args.plan, DEFAULT_PLAN)

    env = EnvConfig.from_env()
    if args.mc_port is not None:
        env.mc_port = args.mc_port
    if args.city:
        env.city = True

    config = JarvisVLAConfig.from_env()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.preview is not None:
        config.preview = args.preview
    if args.async_predict is not None:
        config.async_predict = args.async_predict
    if args.fps is not None:
        config.fps = args.fps

    # Before anything else: a Minecraft boot is ~30s and a world reset another ~10s, and
    # there is no point paying either for a run whose policy is unreachable.
    client = JarvisVLAClient(config)
    print(f"Serving {client.model}.")

    from minestudio.simulator.callbacks import (InitInventoryCallback, RecordCallback,
                                                SummonMobsCallback)

    callbacks = [InitInventoryCallback(INIT_INVENTORY), SummonMobsCallback(MOBS)]
    if args.record:
        # An mp4 per run under logs/, which is how upstream's own evaluation is inspected.
        callbacks.append(RecordCallback(record_path="logs/jarvisvla", fps=20, frame_type="pov"))

    with Session(env) as session:
        sim = session.open(callbacks=callbacks, **sim_kwargs())
        agent = JarvisVLAAgent(sim, config, client=client)

        for number, entry in enumerate(plan, 1):
            print(f"\n[plan] {number}/{len(plan)}: {entry['instruction']!r}", flush=True)
            agent.set_goal(entry["instruction"], entry.get("stop"))

            # Stepping by hand rather than calling run() is the only way to see what the
            # model is pressing while it is pressing it -- which is what tells a stalled run
            # ("the agent stands still") from a lost one ("it is mining the wrong block").
            # The window the agent opens shows the same thing every step; this prints it
            # every --log-every, for the runs that have no display to open one on.
            while agent.busy:
                agent.step()
                if agent.goal.steps % args.log_every == 0:
                    print(f"  step {agent.goal.steps:>3}  {agent.describe_last_action()}", flush=True)

            print(f"[plan] {agent.result}", flush=True)
            if agent.terminated:
                print("[plan] episode ended (death or reset) -- stopping here.")
                break

        agent.close()


if __name__ == "__main__":
    main()
