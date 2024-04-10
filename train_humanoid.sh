#!/bin/sh

env="Humanoid-v4"
save_freq=200000
num_threads=8
seed=1

train()
{
    algo="${1:-sac}"
    echo "env is ${env}, algo is ${algo}, seed is ${seed}"
    export PYTHONPATH=./:$PYTHONPATH
    python ./scripts/train.py --env ${env} --algo ${algo} --save-freq ${save_freq} \
    --num-threads ${num_threads} --seed ${seed} --track
}

train "$@"
