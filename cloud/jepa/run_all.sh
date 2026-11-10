#!/usr/bin/env bash
# W7: second world model (JEPA-style latent dynamics) vs the W4-5 ensemble. All GPU work in one resumable script.
# Run from the repo root. Plan and protocol: experiments/w7_jepa_wm/README.md; setup: cloud/jepa/README.md.
#
#   cloud/jepa/run_all.sh                    # sweep select pick finals   (training + sweep selection)
#   cloud/jepa/run_all.sh report summary probes   # evaluation on the report splits (also runs locally)
#
# Stages
#   sweep    12 latent models, seed 0, + fall-head refit: vicreg x latent dim {64, 256} x window L {1, 3} x
#            prediction weight {25, 100}; ema x d {64, 256} x L {1, 3}
#   select   detector grid of every sweep model on the selection splits (w4_clean, w4_events, w4_events_pits)
#   pick     best sweep configuration by the W5 criterion -> $OUT/sweep_pick.json
#   finals   B seeds 1-2 (winner config), C seeds 0-2 (reconstruction objective, winner d and L),
#            A seeds 1-2 (W4-5 ensemble retrained; seed 0 is the deployed run $A_RUN)
#   report   select + report for A, B, C (all seeds) on the report splits
#   summary  tables, paired bootstrap vs A, figures -> $OUT/results.json
#   probes   friction / ice probes on frozen representations (seed 0 of A, B, C)
# Every stage skips work whose output already exists, so the script can be re-run after an interruption.
set -euo pipefail

PY=${PY:-python}
DATA=${DATA:-logs/wm_data}
OUT=${OUT:-experiments/w7_jepa_wm}
PARALLEL=${PARALLEL:-1}       # concurrent jobs; each holds a ~4 GB split on the GPU plus ~3 GB (d 256): 32 GB -> 3
EPOCHS=${EPOCHS:-20}          # as the W4 ensemble
A_RUN=${A_RUN:-logs/wm/2026-11-07_22-45-27_w4_wm_nominal}
RUNS=${RUNS:-logs/wm}         # where the new run directories go
TRAIN=$DATA/w4_train_nominal  # normal physics only (dynamics); ice and pushes must stay surprising
FALL=$DATA/w4_train           # full randomization (fall heads)
LOGS=$OUT/cloud_logs
mkdir -p "$LOGS" "$OUT/models"
STAGES=${*:-sweep select pick finals}

rundir() { ls -d "$RUNS"/*_"$1" 2>/dev/null | tail -1 || true; }
has() { for s in $STAGES; do [ "$s" = "$1" ] && return 0; done; return 1; }
run_jobs() {  # stdin: one shell command per line; runs $PARALLEL at a time, fails the stage if any job failed
  local cmds=() line running=0 fail=0
  while IFS= read -r line; do [ -n "$line" ] && cmds+=("$line"); done
  for c in "${cmds[@]}"; do
    echo "[run_all]   $c"
    bash -c "$c" &
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then wait -n || fail=1; running=$((running - 1)); fi
  done
  while [ "$running" -gt 0 ]; do wait -n || fail=1; running=$((running - 1)); done
  if [ "$fail" -ne 0 ]; then echo "[run_all] a job failed; see $LOGS/" >&2; return 1; fi
}

latent_job() {  # name reg d L w_pred seed
  local name=$1 reg=$2 d=$3 L=$4 wp=$5 seed=$6
  local r; r=$(rundir "$name")
  if [ -n "$r" ] && [ -f "$r/model_fall.pt" ]; then return; fi
  echo "$PY -m models.world_model_jepa.train --data $TRAIN --name $name --reg $reg --d $d --L $L --w_pred $wp --seed $seed --epochs $EPOCHS --out_root $RUNS > $LOGS/$name.log 2>&1 && \
$PY -m models.world_model_jepa.refit_heads --run \$(ls -d $RUNS/*_$name | tail -1) --data $FALL >> $LOGS/$name.log 2>&1"
}

ensemble_job() {  # name seed
  local name=$1 seed=$2
  local r; r=$(rundir "$name")
  if [ -n "$r" ] && [ -f "$r/model_fall.pt" ]; then return; fi
  echo "$PY -m models.world_model.train --data $TRAIN --name $name --seed $seed --epochs $EPOCHS --out_root $RUNS > $LOGS/$name.log 2>&1 && \
$PY -m models.world_model.refit_fall --run \$(ls -d $RUNS/*_$name | tail -1) --data $FALL >> $LOGS/$name.log 2>&1"
}

select_job() {  # model-name kind checkpoint [extra stages]
  local name=$1 kind=$2 ckpt=$3 stages=${4:-select}
  local need=selection.json; [[ $stages == *report* ]] && need=report.json
  if [ -f "$OUT/models/$name/$need" ]; then return; fi
  echo "$PY -m models.world_model_jepa.compare --stage $stages --model '$name=$kind:$ckpt' --data $DATA --train_split $TRAIN --out $OUT > $LOGS/compare_$name.log 2>&1"
}

SWEEP=()
if [ -n "${SWEEP_CONFIGS:-}" ]; then  # "reg d L w_pred;...", e.g. "vicreg 64 1 25;ema 64 1 25" (smoke tests)
  IFS=';' read -ra SWEEP <<< "$SWEEP_CONFIGS"
else
  # vicreg: the prediction weight trades predictability against keeping information (w_var 25, w_cov 1).
  # 20k-iteration pilots (d 128, L 3): w_pred 5 -> key detection 0.21, 25 -> 0.29, so the sweep goes up, not down.
  # ema has no such term.
  for d in 64 256; do for L in 1 3; do for wp in 25 100; do SWEEP+=("vicreg $d $L $wp"); done; done; done
  for d in 64 256; do for L in 1 3; do SWEEP+=("ema $d $L 25"); done; done
fi
sname() { echo "$1_d$2_L$3_wp$4"; }

if has sweep; then
  echo "[run_all] sweep: ${#SWEEP[@]} latent models, $PARALLEL at a time"
  for c in "${SWEEP[@]}"; do set -- $c; latent_job "w7_Bs_$(sname $@)" "$1" "$2" "$3" "$4" 0; done | run_jobs
fi

if has select; then
  echo "[run_all] select: detector grid per sweep model"
  for c in "${SWEEP[@]}"; do set -- $c; select_job "Bs_$(sname $@)" latent "$(rundir "w7_Bs_$(sname $@)")/model_fall.pt"; done | run_jobs
fi

if has pick; then
  $PY -m models.world_model_jepa.sweep_pick --out "$OUT" --prefix Bs_ | tee "$OUT/sweep_pick.txt" | tail -1 > "$OUT/sweep_pick.json"
  echo "[run_all] winner: $(cat "$OUT/sweep_pick.json")"
fi

best() { $PY -c "import json; print(json.load(open('$OUT/sweep_pick.json'))['$1'])"; }

if has finals; then
  REG=$(best reg); D=$(best d); L=$(best L); WP=$(best w_pred)
  echo "[run_all] finals: B ($REG d $D L $L w_pred $WP) seeds 1-2, C (recon, same d L w_pred) seeds 0-2, A seeds 1-2"
  {
    for s in 1 2; do latent_job "w7_B_s$s" "$REG" "$D" "$L" "$WP" "$s"; done
    for s in 0 1 2; do latent_job "w7_C_s$s" recon "$D" "$L" "$WP" "$s"; done
    for s in 1 2; do ensemble_job "w7_A_s$s" "$s"; done
  } | run_jobs
fi

if has report; then
  B0=$(best checkpoint)
  echo "[run_all] report: A, B, C on the report splits"
  {
    select_job A ensemble "$A_RUN/model_fall.pt" "select report"
    for s in 1 2; do select_job "A@s$s" ensemble "$(rundir "w7_A_s$s")/model_fall.pt" "select report"; done
    select_job B latent "$B0" "select report"
    for s in 1 2; do select_job "B@s$s" latent "$(rundir "w7_B_s$s")/model_fall.pt" "select report"; done
    select_job C latent "$(rundir w7_C_s0)/model_fall.pt" "select report"
    for s in 1 2; do select_job "C@s$s" latent "$(rundir "w7_C_s$s")/model_fall.pt" "select report"; done
  } | run_jobs
fi

if has summary; then
  $PY -m models.world_model_jepa.compare --stage summary --baseline A --out "$OUT" | tee "$LOGS/summary.log"
fi

if has probes; then
  $PY -m models.world_model_jepa.probes --data "$FALL" --out "$OUT" --model "A=ensemble:$A_RUN/model_fall.pt" \
    --model "B=latent:$(best checkpoint)" --model "C=latent:$(rundir w7_C_s0)/model_fall.pt" | tee "$LOGS/probes.log"
fi
echo "[run_all] done: $STAGES"
