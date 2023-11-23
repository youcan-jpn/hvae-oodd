#!/bin/bash
#SBATCH --mail-type=ALL # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/yutono/net_scratch/hvae-oodd/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/yutono/net_scratch/hvae-oodd/jobs/%j.err # where to store error messages
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclude=tikgpu[08-10]
#CommentSBATCH --nodelist=tikgpu07 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'



ETH_USERNAME=yutono
PROJECT_NAME=hvae-oodd
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=my-oodd
mkdir -p ${DIRECTORY}/jobs
#TODO: change your ETH USERNAME and other stuff from above according + in the #SBATCH output and error the path needs to be double checked!

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"


[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

# Execute your code
python scripts/dvae_run.py \
--epochs 500 \
--batch_size 32 \
--free_nats 2 \
--free_nats_epochs 100 \
--warmup_epochs 50 \
--test_every 10 \
--train_datasets '{ "CIFAR10EmbeddingSimCLR": {"split": "train"}}' \
--val_datasets \
'{
    "CIFAR10EmbeddingSimCLR": {"split": "validation"}
}' \
--model VAE \
--likelihood DiscretizedLogisticMixLikelihoodConv2d \
--config_deterministic \
'[
    [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 5, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 2, "weightnorm": true, "gated": false}
    ]
]' \
--config_stochastic \
'[
    {"block": "GaussianConv2d", "latent_features": 4, "weightnorm": true},
    {"block": "GaussianConv2d", "latent_features": 4, "weightnorm": true},
    {"block": "GaussianDense", "latent_features": 32, "weightnorm": true}
]'

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
