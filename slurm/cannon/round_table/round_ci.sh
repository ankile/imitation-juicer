#!/bin/bash

#SBATCH -p gpu                
#SBATCH -t 1-12:00            
#SBATCH --mem=256G            
#SBATCH --gres=gpu:1          
#SBATCH -c 16               
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log   

python -m src.train.bc_no_rollout \
    +experiment=image_collect_infer \
    furniture=round_table \
    data.data_subset=50 \
    data.dataloader_workers=16
