#!/bin/zsh
# waits for the data fetch, smoke-tests, then runs the full local training. Logs to work/run_all.log
cd /Users/maninij/TruthLens/image_retrain
PY=../.venv/bin/python
echo "[$(date)] waiting for local_manifest.csv"
while [ ! -f local_manifest.csv ]; do sleep 20; done
echo "[$(date)] fetch done: $(wc -l < local_manifest.csv) rows"
rm -rf work/run_local
echo "[$(date)] smoke test"
SMOKE=1 $PY train_local.py > work/smoke.log 2>&1 || { echo "[$(date)] SMOKE FAILED"; tail -20 work/smoke.log; exit 1; }
echo "[$(date)] smoke ok"
rm -rf work/run_local
echo "[$(date)] full run"
$PY train_local.py > work/full.log 2>&1 || { echo "[$(date)] FULL RUN FAILED"; tail -20 work/full.log; exit 1; }
echo "[$(date)] FULL RUN DONE"
