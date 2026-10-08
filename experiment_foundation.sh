#!/bin/bash

for D in \
    "gas-drift"
do
    for S in 11 22 33 44 55
    do
        uv run python skeleton/experiment.py \
            --seed "$S" \
            --methods foundation \
            --profile "exp" \
            --context 1000 \
            --dataset "$D"
    done
done

# run this onece: chmod +x experiment_foundation.sh
# then run: ./experiment_foundation.sh