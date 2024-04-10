#!/bin/sh

env="Humanoid-v4"
log_folder=./logs/
exp_id=0

eval()
{
    algo="${1:-sac}"
    export PYTHONPATH=./:$PYTHONPATH
    python ./scripts/eval.py --env ${env} --algo ${algo} -f ${log_folder} --exp-id ${exp_id} --load-best
}

eval "$@"

