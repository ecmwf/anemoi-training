#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --job-name=aifs-dowa-model-test

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# generic settings
CONDA_ENV=aifs
GITDIR=/hpcperm/nld1247/aifs-lam-dowa
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
# srun python train_from_checkpoint.py
srun anemoi-training train --config-name=stretched_grid.yaml