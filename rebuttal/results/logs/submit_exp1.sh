#!/bin/bash
# Submit the Exp 1 kappa_l ablation matrix: ALGO x KL x SEED.
# Usage: bash submit_exp1.sh            # full matrix (12 runs)
#        bash submit_exp1.sh dry        # print sbatch lines only, don't submit
DRY="$1"
REPO=/storage/home/impwxu/code/EPO
LOGDIR=$REPO/rebuttal/results/exp1
mkdir -p "$LOGDIR"
export GIT_AUTHOR_NAME="Wujiang Xu" GIT_AUTHOR_EMAIL="impwxu@fb.com"

for ALGO in ppo grpo; do
  for KL in 0 0.5 0.8; do
    for SEED in 0 1; do
      klstr=${KL/./p}
      NAME="exp1_${ALGO}_kl${klstr}_s${SEED}"
      OUT="$LOGDIR/${NAME}.out"
      if [ "$DRY" = "dry" ]; then
        echo "sbatch -J $NAME -o $OUT -e $OUT --export=ALL,ALGO=$ALGO,KL=$KL,SEED=$SEED $REPO/rebuttal/results/logs/run_exp1.sbatch"
      else
        JID=$(sbatch --parsable -J "$NAME" -o "$OUT" -e "$OUT" \
              --export=ALL,ALGO=$ALGO,KL=$KL,SEED=$SEED \
              "$REPO/rebuttal/results/logs/run_exp1.sbatch")
        echo "$NAME -> job $JID"
        echo "$JID $NAME ALGO=$ALGO KL=$KL SEED=$SEED $(date)" >> "$LOGDIR/submitted.txt"
      fi
    done
  done
done
