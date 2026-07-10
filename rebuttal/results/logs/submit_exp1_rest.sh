#!/bin/bash
# Submit the REMAINING 10 Exp 1 runs (everything except the 2 pilots ppo/0/0 and ppo/0.8/0).
REPO=/storage/home/impwxu/code/EPO
LOGDIR=$REPO/rebuttal/results/exp1
RS=$REPO/rebuttal/results/logs/run_exp1.sbatch
mkdir -p "$LOGDIR"
skip="ppo|0|0 ppo|0.8|0"   # already submitted as pilots
for ALGO in ppo grpo; do
  for KL in 0 0.5 0.8; do
    for SEED in 0 1; do
      key="$ALGO|$KL|$SEED"
      case " $skip " in *" $key "*) echo "skip pilot $key"; continue;; esac
      klstr=${KL/./p}; NAME="exp1_${ALGO}_kl${klstr}_s${SEED}"; OUT="$LOGDIR/${NAME}.out"
      JID=$(sbatch --parsable -J "$NAME" -o "$OUT" -e "$OUT" \
            --export=ALL,ALGO=$ALGO,KL=$KL,SEED=$SEED "$RS")
      echo "$NAME -> job $JID"
      echo "$JID $NAME ALGO=$ALGO KL=$KL SEED=$SEED $(date)" >> "$LOGDIR/submitted.txt"
    done
  done
done
