#!/bin/sh

env="Humanoid-v4"
algo="sac"
log_folder=./logs/
exp_id=0

export PYTHONPATH=./:$PYTHONPATH
python ./scripts/eval.py --env ${env} --algo ${algo} -f ${log_folder} --exp-id ${exp_id} --load-best


