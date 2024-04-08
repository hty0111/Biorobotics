#!/bin/sh

env="Humanoid-v4"
algo="sac"
num_steps=2000000
save_freq=100000
num_threads=8
seed=1

echo "env is ${env}, algo is ${algo}, seed is ${seed}"
export PYTHONPATH=./:$PYTHONPATH
python ./scripts/train.py --env ${env} --algo ${algo} -n ${num_steps} --save-freq ${save_freq} \
--num-threads ${num_threads} --seed ${seed}


