#!/bin/zsh
# Waits for Kaggle's weekly GPU quota to reset, pushes the photo notebook, watches it, downloads the outputs. Log: work/kaggle_photo_r1.log
# Retries the push every 10 min for up to 10 h. Does nothing else (no deletes, no other Kaggle actions).
cd /Users/maninij/TruthLens/image_retrain || exit 1
K=/Users/maninij/TruthLens/video_retrain/work/kvenv/bin/kaggle
LOG=work/kaggle_photo_r1.log; OUT=work/kaggle_photo_r1_out
say() { echo "[$(date '+%F %T')] $*" >> $LOG; }
say "started; waiting for GPU quota"
for i in {1..60}; do
  r=$($K kernels push -p kaggle_photo_r1 2>&1); say "push attempt $i: $r"
  echo "$r" | grep -qi "successfully pushed" && break
  sleep 240
done
echo "$r" | grep -qi "successfully pushed" || { say "gave up: push never accepted"; exit 1; }
say "pushed; polling status"
for i in {1..80}; do
  sleep 300
  s=$($K kernels status maninijaiswal/truthlens-photo-swap-r1 2>&1); say "status: $s"
  echo "$s" | grep -qi "COMPLETE\|ERROR\|CANCEL" && break
done
mkdir -p $OUT; $K kernels output maninijaiswal/truthlens-photo-swap-r1 -p $OUT >> $LOG 2>&1
say "finished; outputs (if any) in $OUT"
