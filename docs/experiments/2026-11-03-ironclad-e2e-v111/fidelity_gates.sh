#!/usr/bin/env bash
# F2-F6 and the F9 soak for the e2e Ironclad campaign (plan §3.5, WP8).
#
# Runs every gate in turn at nice 10 with $J workers. Before each one it waits
# while a DEV baseline (scripts/e2e_arm.sh) runs, so the gates never share the
# box with one. Full output goes to $OUT/<gate>.log; $OUT/gates.log holds the
# start, end and exit code of each. FIDELITY.md reads the results.
#
# Usage: fidelity_gates.sh [gate ...]   (default: all, in the order below)
set -uo pipefail

PYCLI="$(cd "$(dirname "$0")/../../../sts2-pycli" && pwd)"
PY="${E2E_PY:-$HOME/projects/sts2-sim-e2e/sts2-pycli/.venv/bin/python}"
OUT="${E2E_FIDELITY_OUT:-$HOME/runs/e2e/fidelity}"
AD="${E2E_AD_REPO:-$HOME/runs/e2e/ad-36bd1fd}"
J="${E2E_FIDELITY_JOBS:-16}"
ELITES=DECIMILLIPEDE_ELITE,PHROG_PARASITE_ELITE,KNIGHTS_ELITE,SOUL_NEXUS_ELITE,MECHA_KNIGHT_ELITE
BOSSES=CEREMONIAL_BEAST_BOSS,THE_KIN_BOSS,VANTOM_BOSS,LAGAVULIN_MATRIARCH_BOSS,SOUL_FYSH_BOSS
BOSSES=$BOSSES,WATERFALL_GIANT_BOSS,KAISER_CRAB_BOSS,KNOWLEDGE_DEMON_BOSS,THE_INSATIABLE_BOSS
BOSSES=$BOSSES,AEONGLASS_BOSS,QUEEN_BOSS,TEST_SUBJECT_BOSS
mkdir -p "$OUT"
cd "$PYCLI" || exit 1

wait_free() {
    # The bracket keeps pgrep from matching this shell's own command line.
    while pgrep -f '[s]cripts/e2e_arm\.sh' > /dev/null; do sleep 60; done
}

gate() {
    local name=$1
    shift
    wait_free
    echo "== $name start $(date '+%F %T') HEAD $(git rev-parse --short HEAD)" | tee -a "$OUT/gates.log"
    { time env PYTHONPATH=python nice -n 10 "$@"; } > "$OUT/$name.log" 2>&1
    local rc=$?
    echo "== $name exit $rc $(date '+%F %T')" | tee -a "$OUT/gates.log"
}

identity() {
    {
        echo "== engine identity $(date '+%F %T')"
        PYTHONPATH=python "$PY" -m ppo.wheel_guard
        PYTHONPATH=python "$PY" -c 'import json, sts2_rs; print(json.dumps(sts2_rs.build_info(), sort_keys=True))'
        sha256sum "$("$PY" -c 'import sts2_rs.sts2_rs as m; print(m.__file__)')"
    } >> "$OUT/identity.log" 2>&1
}

run_gate() {
    case $1 in
    f3_qx) gate f3_qx "$PY" python/run_parity.py --sweep --game-version v0.111.0 \
        --character Ironclad --ascension 10 --seed QX --seeds 200 -j "$J" \
        --cache-dir "$OUT/run_parity_cache" --out "$OUT/f3_qx" ;;
    f3_qx_rerun) gate f3_qx_rerun "$PY" python/run_parity.py --sweep --game-version v0.111.0 \
        --character Ironclad --ascension 10 --seed QX --seeds 200 -j "$J" \
        --cache-dir "$OUT/run_parity_cache_rerun" --out "$OUT/f3_qx_rerun" ;;
    f3_fresh) gate f3_fresh "$PY" python/run_parity.py --sweep --game-version v0.111.0 \
        --character Ironclad --ascension 10 --seed E2EF --seeds 200 -j "$J" \
        --cache-dir "$OUT/run_parity_cache" --out "$OUT/f3_fresh" ;;
    f5_v1) gate f5_v1 "$PY" -m ppo.e2e.fidelity_parity encounters --encounters "$ELITES" \
        --trials 60 --seed-start 0 --max-turns 15 --pool v1 -j "$J" --out "$OUT/f5_v1.json" ;;
    f5_v2) gate f5_v2 "$PY" -m ppo.e2e.fidelity_parity encounters --encounters "$ELITES" \
        --trials 60 --seed-start 0 --max-turns 15 --pool v2 -j "$J" --out "$OUT/f5_v2.json" ;;
    f5_relics) gate f5_relics "$PY" -m ppo.e2e.fidelity_parity encounters \
        --encounters "$ELITES" --trials 40 --seed-start 1000 --max-turns 15 --pool v1 \
        --relics FESTIVE_POPPER,MERCURY_HOURGLASS,BRONZE_SCALES,LETTER_OPENER \
        -j "$J" --out "$OUT/f5_relics.json" ;;
    f5_bosses_v1) gate f5_bosses_v1 "$PY" -m ppo.e2e.fidelity_parity encounters \
        --encounters "$BOSSES" --trials 25 --seed-start 0 --max-turns 15 --pool v1 \
        -j "$J" --out "$OUT/f5_bosses_v1.json" ;;
    f5_bosses_v2) gate f5_bosses_v2 "$PY" -m ppo.e2e.fidelity_parity encounters \
        --encounters "$BOSSES" --trials 25 --seed-start 0 --max-turns 15 --pool v2 \
        -j "$J" --out "$OUT/f5_bosses_v2.json" ;;
    f6) gate f6 "$PY" -m ppo.e2e.fidelity_parity random-target --trials 600 --seed-start 0 \
        --max-turns 12 -j "$J" --out "$OUT/f6_random_target.json" ;;
    f6_aligned) gate f6_aligned "$PY" -m ppo.e2e.fidelity_parity random-target --trials 600 \
        --seed-start 0 --max-turns 12 --align-setup-deaths -j "$J" \
        --out "$OUT/f6_aligned.json" ;;
    f4_a10_act3) gate f4_a10_act3 "$PY" -m ppo.e2e.act23_a10 --character Ironclad \
        --ascension 10 --game-version v0.111.0 --full-state --trace-split -n 50 --workers "$J" ;;
    f4_a10) gate f4_a10 "$PY" python/act23_state_parity.py --character Ironclad \
        --ascension 10 --game-version v0.111.0 --full-state --trace-split -n 50 --workers "$J" ;;
    f4_a0_ic) gate f4_a0_ic "$PY" python/act23_state_parity.py --character Ironclad \
        --ascension 0 --game-version v0.111.0 --full-state --trace-split -n 50 --workers "$J" ;;
    f4_a0_all) gate f4_a0_all "$PY" python/act23_state_parity.py --all-characters \
        --ascension 0 --game-version v0.111.0 --full-state --trace-split -n 20 --workers "$J" ;;
    f2) gate f2 "$PY" experiments/alphadefect-compare/replay_parity.py --repo "$AD" \
        --out "$OUT/f2_replay_parity.jsonl" --procs "$J"
        "$PY" experiments/alphadefect-compare/bucket.py "$OUT/f2_replay_parity.jsonl" 200 \
            > "$OUT/f2_buckets.txt" 2>&1 ;;
    f9_soak) gate f9_soak "$PY" -m ppo.e2e.invariant_soak --runs 4000 --workers "$J" \
        --max-steps 6000 --out "$OUT/f9_soak.json" ;;
    *) echo "unknown gate $1" >&2; return 2 ;;
    esac
}

identity
GATES=("$@")
if [ ${#GATES[@]} -eq 0 ]; then
    GATES=(f3_qx f3_fresh f5_v1 f5_v2 f5_relics f5_bosses_v1 f5_bosses_v2 f6
        f4_a10 f4_a0_ic f4_a0_all f2 f9_soak f3_qx_rerun f6_aligned f4_a10_act3)
fi
for g in "${GATES[@]}"; do
    run_gate "$g"
done
identity
echo "== all done $(date '+%F %T')" | tee -a "$OUT/gates.log"
