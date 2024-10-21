#!/bin/bash
#SBATCH --output=/scratch/project_465001383/aifs/logs/name2.out
#SBATCH --error=/scratch/project_465001383/aifs/logs/name2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=project_465001383
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00
#SBATCH --job-name=aifs
#SBATCH --exclusive

PROJECT_DIR=/pfs/lustrep4/scratch/$SLURM_JOB_ACCOUNT
CONTAINER_SCRIPT=$PROJECT_DIR/aifs/run-pytorch/run-pytorch.sh

#CHANGE THESE:
CONTAINER=$PROJECT_DIR/aifs/container/containers/aifs-met-pytorch-2.2.0-rocm-5.6.1-py3.9-v2.0-new-correct-anemoi-models-sort-vars.sif
PYTHON_SCRIPT=$PROJECT_DIR/haugenha/anemoi-training-setup/anemoi-training-config/anemoi-training/src/lumi_train.py
VENV=/users/haugenha/work/.venv-anemoi-training


module load LUMI/23.09 partition/G

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:${SINGULARITYENV_LD_LIBRARY_PATH}

# MPI + OpenMP bindings: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"

if [[ "$VENV" != "None" && -n "$VENV" ]]; then
# Set this virtual environment
    export VIRTUAL_ENV=$VENV

# Ensure the virtual environment is loaded inside the container
    export PYTHONUSERBASE=$VIRTUAL_ENV
    export PATH=$PATH:$VIRTUAL_ENV/bin
else
    :
fi

# run run-pytorch.sh in singularity container like recommended
# in LUMI doc: https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch
srun --cpu-bind=$CPU_BIND \
    singularity exec -B /pfs:/pfs \
                     -B /var/spool/slurmd \
                     -B /opt/cray \
                     -B /usr/lib64 \
                     -B /usr/lib64/libjansson.so.4 \
        $CONTAINER $CONTAINER_SCRIPT $PYTHON_SCRIPT