#!/bin/bash

# Make a list of furniture types
furniture_types=(
    "one_leg"
    "round_table"
    "lamp"
    "square_table"
)

# Infinite loop
while true; do
    # Loop through each project-furniture pair
    for furniture in "${furniture_types[@]}"; do
        # Construct and execute the command
        command="python -m src.eval.evaluate_model --n-envs 10 --n-rollouts 10 --randomness low -f $furniture --project-id multitask-everything-1 --wandb --if-exists append --action-type pos --run-state finished --prioritize-fewest-rollout --max-rollouts 100 --multitask"
        echo "Executing: $command"
        
        eval $command
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "Execution for $furniture in $project_id completed successfully."
        else
            echo "Error encountered during execution for $project_id. Check ${project_id}_errors.log for details."
        fi
    done

done