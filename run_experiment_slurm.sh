#!/bin/bash

#SBATCH --job-name="DiPrivML"
#SBATCH --ntasks=50

if [[ $# -eq 0 ]] ; then # if called with no arguments
    echo "Usage: sbatch $0 <DATASET NAME> <CONDA ENV>"
    exit 1
fi

DATASET=$1
ATTACK_PY=attack.py

source /etc/profile.d/modules.sh
module load anaconda3
module load parallel
source activate $2

# flags for compatibility between tensorflow and theano
export KMP_DUPLICATE_LIB_OK=TRUE
export THEANO_FLAGS=device=cpu

python attack.py $DATASET --save_data=1

# `parallel` will run the quoted command, replacing each {n} with each value on
# the n-th line starting with ":::" exactly as if it were a nested for-loop.
# The start-times, durations, and commands of tasks are stored in joblog.txt.
parallel -j $SLURM_NTASKS --joblog joblog.txt \
    "srun --exclusive -N1 -n1 python $ATTACK_PY $DATASET \
    --target_model={1} \
    --target_l2_ratio={2} \
    --target_privacy='grad_pert' \
    --target_dp={3} \
    --target_epsilon={4} \
    --run={5}" \
    ::: 'softmax' 'nn' \
    :::+ '1e-5' '1e-4' \
    # '+' means we get (softmax, 1e-5) and (nn, 1e-4) vs. all combinations
    ::: 'dp' 'adv_cmp' 'rdp' 'zcdp' \
    ::: 0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0 100.0 500.0 1000.0 \
    ::: 1 2 3 4 5
echo done
