"""Feasibility spike for the benchmark referee (Spire Bench, milestone 0).

Plays full runs on the real game DLL with the keyed RNG (``STS2_REFEREE=1``,
``src/Sts2Headless/RefereeRng.cs``) and a trivial policy, and measures what the
go / no-go decision needs:

* throughput (decisions per second, runs per hour) and peak memory per process;
* RNG escapes (``referee_fault``) and world errors on stderr;
* determinism: the same key and the same actions reproduce the same game;
* key dependence: a different key gives a different game.

Every finished run is appended to the output JSONL at once, so a stuck or
failing sweep is visible while it runs.

Usage::

    python python/referee_spike.py --runs 20 --ascension 10 --out /tmp/spike.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import sts2_client

CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]
MAX_STEPS = 20000
STUCK_LIMIT = 20


def _act(action: str, **args: Any) -> dict[str, Any]:
    cmd: dict[str, Any] = {"cmd": "action", "action": action}
    if args:
        cmd["args"] = args
    return cmd


def choose(state: dict[str, Any], rnd: random.Random) -> dict[str, Any]:
    """The trivial policy of ``play_full_run.py``, as one decision -> one command."""
    decision = state.get("decision", "")
    if decision == "map_select":
        choice = rnd.choice(state["choices"])
        return _act("select_map_node", col=choice["col"], row=choice["row"])
    if decision == "combat_play":
        energy = state.get("energy", 0)
        playable = [
            c for c in state.get("hand", []) if c.get("can_play") and c.get("cost", 0) <= energy
        ]
        if not playable:
            return _act("end_turn")
        card = playable[0]
        if card.get("target_type") == "AnyEnemy" and state.get("enemies"):
            return _act("play_card", card_index=card["index"], target_index=0)
        return _act("play_card", card_index=card["index"])
    if decision in ("event_choice", "rest_site"):
        options = state.get("options", [])
        if decision == "rest_site":
            options = [o for o in options if o.get("is_enabled", True)]
            options.sort(key=lambda o: o.get("option_id") != "HEAL")
        else:
            options = [o for o in options if not o.get("is_locked")] or options
        if not options:
            return _act("leave_room")
        return _act("choose_option", option_index=options[0]["index"])
    if decision == "card_reward":
        if state.get("cards"):
            return _act("select_card_reward", card_index=0)
        return _act("skip_card_reward")
    if decision == "bundle_select":
        return _act("select_bundle", bundle_index=0)
    if decision == "card_select":
        if state.get("cards"):
            return _act("select_cards", indices="0")
        return _act("skip_select")
    if decision == "shop":
        return _act("leave_room")
    if decision == "potion_swap":
        return _act("potion_swap")  # no discard_slot = decline the potion
    if decision == "rewards" and state.get("rewards"):
        # Take everything, as the host used to (GH #1282).
        return _act("take_reward", reward_index=state["rewards"][0]["index"])
    return _act("proceed")


def choose_legal(state: dict[str, Any], rnd: random.Random) -> dict[str, Any] | None:
    """A random legal action from a referee observation, or None if it lists none.

    In combat it mostly plays or drinks rather than ending the turn at once, so
    fights last long enough to exercise the game.
    """
    legal = state.get("legal_actions") or []
    if not legal:
        return None
    if state.get("decision") == "combat_play":
        active = [a for a in legal if a["action"] in ("play_card", "use_potion")]
        if active and rnd.random() < 0.9:
            legal = active
        else:
            legal = [a for a in legal if a["action"] == "end_turn"] or legal
    pick = rnd.choice(legal)
    cmd: dict[str, Any] = {"cmd": "action", "action": pick["action"]}
    if "choose" in pick:
        spec = pick["choose"]
        count = rnd.randint(spec["min"], spec["max"])
        cmd["args"] = {"indices": ",".join(str(i) for i in rnd.sample(spec["from"], count))}
    elif "args" in pick:
        cmd["args"] = pick["args"]
    return cmd


def _peak_rss_mb(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def play(
    *,
    character: str,
    ascension: int,
    label: str,
    key_hex: str | None,
    actions: list[dict[str, Any]] | None = None,
    census: bool = False,
) -> dict[str, Any]:
    """Plays one run. ``key_hex`` None = the game's own RNG (no referee mode).

    ``actions`` replays a recorded command list instead of asking the policy.
    ``census`` reads the host's swallowed-error counts at the end
    (``host_faults``, GH #1283) into ``result["host_faults"]``.
    """
    env = os.environ.copy()
    env.pop("STS2_REFEREE", None)
    if key_hex is not None:
        env["STS2_REFEREE"] = "1"
        env["STS2_REFEREE_DIAG"] = "1"  # this driver reads referee_stats
    rnd = random.Random(label)
    sent: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    result: dict[str, Any] = {
        "character": character,
        "ascension": ascension,
        "label": label,
        "referee": key_hex is not None,
        "replay": actions is not None,
    }
    with tempfile.TemporaryFile("w+", encoding="utf-8") as err:
        proc = subprocess.Popen(
            sts2_client.headless_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            bufsize=1,
            env=env,
            cwd=sts2_client.REPO_ROOT,
        )
        assert proc.stdin is not None and proc.stdout is not None
        stdin, stdout = proc.stdin, proc.stdout

        def read() -> dict[str, Any]:
            while True:
                line = stdout.readline()
                if not line:
                    raise RuntimeError("EOF from game process")
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)

        def send(cmd: dict[str, Any], *, record: bool = True) -> dict[str, Any]:
            stdin.write(json.dumps(cmd) + "\n")
            stdin.flush()
            reply = read()
            if record:
                sent.append(cmd)
                digest.update(json.dumps(reply, sort_keys=True).encode())
            return reply

        t0 = time.perf_counter()
        try:
            ready = read()
            result["boot_s"] = round(time.perf_counter() - t0, 3)
            if ready.get("referee", False) != (key_hex is not None):
                raise RuntimeError(f"host referee flag mismatch: {ready}")
            start: dict[str, Any] = {
                "cmd": "start_run",
                "character": character,
                "seed": label,
                "ascension": ascension,
                "act1": "roll",
            }
            if key_hex is not None:
                start["rng_key"] = key_hex
            t1 = time.perf_counter()
            state = send(start)
            steps = stuck = 0
            last_key = None
            decided = state
            queue = list(actions) if actions is not None else None
            while steps < MAX_STEPS:
                kind = state.get("type")
                if kind == "referee_fault":
                    result["fault"] = state.get("message")
                    break
                if state.get("decision") == "game_over":
                    result["victory"] = bool(state.get("victory"))
                    break
                if queue is not None:
                    if not queue:
                        result["error"] = "replay ran out of actions"
                        break
                    cmd = queue.pop(0)
                else:
                    if kind == "error":
                        result["errors"] = result.get("errors", 0) + 1
                        code = state.get("code", "none")
                        result.setdefault("error_codes", {}).setdefault(code, 0)
                        result["error_codes"][code] += 1
                        if key_hex is not None:
                            # The decision stands; the listed action was refused.
                            result.setdefault("refused", []).append(
                                {"decision": decided.get("decision"), "cmd": sent[-1]}
                            )
                            state = decided
                    else:
                        decided = state
                    if key_hex is not None:
                        cmd = choose_legal(state, rnd)
                        if cmd is None:
                            result["error"] = f"no legal action at {state.get('decision')}"
                            break
                    elif kind == "error":
                        cmd = _act("proceed")
                    else:
                        cmd = choose(state, rnd)
                    player = state.get("player") or {}
                    key = (
                        state.get("decision"),
                        state.get("round"),
                        player.get("hp"),
                        len(state.get("hand", [])),
                        sum(e.get("hp", 0) for e in state.get("enemies", [])),
                        state.get("energy"),
                        state.get("floor"),
                    )
                    stuck = stuck + 1 if key == last_key else 0
                    last_key = key
                    if stuck > STUCK_LIMIT:
                        result["error"] = f"stuck at {state.get('decision')}"
                        break
                state = send(cmd)
                steps += 1
            else:
                result["error"] = "step limit"
            elapsed = time.perf_counter() - t1
            result.update(
                steps=steps,
                play_s=round(elapsed, 3),
                steps_per_s=round(steps / elapsed, 1) if elapsed > 0 else None,
                act=state.get("act"),
                floor=state.get("floor"),
                digest=digest.hexdigest(),
                peak_rss_mb=_peak_rss_mb(proc.pid),
            )
            if census and "fault" not in result:
                faults = send({"cmd": "host_faults"}, record=False)
                result["host_faults"] = faults.get("counts", {})
                result["host_fault_samples"] = faults.get("samples", {})
            if key_hex is not None and "fault" not in result:
                stats = send({"cmd": "referee_stats"}, record=False)
                result["draws"] = stats.get("draws")
                result["unclassified_fields"] = stats.get("unclassified_fields", [])
        except Exception as exc:  # one bad run must not stop the sweep
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                stdin.write('{"cmd":"quit"}\n')
                stdin.flush()
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
        err.seek(0)
        world = [ln.rstrip() for ln in err if sts2_client.is_world_error_line(ln)]
        result["world_errors"] = len(world)
        if world:
            result["world_error_sample"] = world[:3]
    result["actions"] = sent[1:]
    return result


def episode_key(master: bytes, label: str) -> str:
    return hmac.new(master, label.encode(), hashlib.sha256).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Feasibility spike for the benchmark referee.")
    ap.add_argument("--runs", type=int, default=5, help="runs per character")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--characters", nargs="+", default=CHARACTERS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--replays", type=int, default=2, help="runs per character to replay-check")
    ap.add_argument("--vanilla", action="store_true", help="game RNG instead of the keyed one")
    ap.add_argument("--master", default="spire-bench-spike", help="master secret for the spike")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    master = args.master.encode()
    jobs = [(ch, f"SPIKE{i}") for ch in args.characters for i in range(args.runs)]

    def first(job: tuple[str, str]) -> list[dict[str, Any]]:
        ch, label = job
        key = None if args.vanilla else episode_key(master, f"{ch}:{label}")
        run = play(character=ch, ascension=args.ascension, label=label, key_hex=key)
        out = [run]
        index = int(label.removeprefix("SPIKE"))
        if key is not None and index < args.replays and "error" not in run and "fault" not in run:
            again = play(
                character=ch,
                ascension=args.ascension,
                label=label,
                key_hex=key,
                actions=run["actions"],
            )
            again["replay_matches"] = again.get("digest") == run["digest"]
            other = play(
                character=ch,
                ascension=args.ascension,
                label=label,
                key_hex=episode_key(master, f"{ch}:{label}:other"),
            )
            other["other_key"] = True
            other["differs_from_first"] = other.get("digest") != run["digest"]
            out += [again, other]
        return out

    done = 0
    with open(args.out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        t0 = time.perf_counter()
        for fut in as_completed([pool.submit(first, j) for j in jobs]):
            for row in fut.result():
                row = {k: v for k, v in row.items() if k != "actions"}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            done += 1
            print(
                f"[{done}/{len(jobs)}] {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
