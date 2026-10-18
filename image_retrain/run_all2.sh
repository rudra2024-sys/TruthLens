#!/bin/zsh
# round 1b: wait for extra real-only data, smoke-test, then full run. Logs to work/run_all2.log
cd /Users/maninij/TruthLens/image_retrain
PY=../.venv/bin/python
echo "[$(date)] waiting for local_manifest_extra.csv"
while [ ! -f local_manifest_extra.csv ]; do sleep 20; done
echo "[$(date)] extras done: $(wc -l < local_manifest_extra.csv) rows"
rm -rf work/run_local_r2
SMOKE=1 $PY train_local_r2.py > work/smoke2.log 2>&1 || { echo "[$(date)] SMOKE FAILED"; tail -25 work/smoke2.log; exit 1; }
echo "[$(date)] smoke ok"
rm -rf work/run_local_r2
$PY train_local_r2.py > work/full2.log 2>&1 || { echo "[$(date)] FULL RUN FAILED"; tail -25 work/full2.log; exit 1; }
echo "[$(date)] FULL RUN DONE"
