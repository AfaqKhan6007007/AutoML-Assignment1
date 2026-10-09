#!/bin/bash

for D in \
    "breast-w" \
    "credit-g" \
    "phoneme" \
    "Phishing_Legitimate_full" \
    "gas-drift"
do
    for S in 11 22 33 44 55
    do
        uv run python skeleton/experiment.py \
            --seed "$S" \
            --methods default random smbo hyperband foundation \
            --profile "exp" \
            --dataset "$D"
    done
done

# run this onece: chmod +x experiment.sh
# then run: ./experiment.sh
