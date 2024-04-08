#!/bin/sh

env="Humanoid-v4"
algo="sac"
num_steps=2000000
save_freq=100000
seed=1

export PYTHONPATH=./:$PYTHONPATH
python ./scripts/train.py --env ${env} --algo ${algo} -n ${num_steps} --save-freq ${save_freq} --seed ${seed}


