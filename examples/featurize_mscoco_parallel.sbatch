#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-19
#SBATCH --job-name=feat
#SBATCH --output=logs/featurize_logs_mscoco_%a_%j.log

MODEL_ID=$SLURM_ARRAY_TASK_ID
echo $MODEL_ID

python examples/featurize_mscoco_parallel.py --ckpt_dir <CHECKPOINTS DIR> \
                                              --save_dir <DIR WHERE TRAK FEATURES WILL BE SAVED>  \
                                              --model_id $MODEL_ID
