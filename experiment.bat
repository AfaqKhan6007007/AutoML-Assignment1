@echo off
for %%D in (
    "Phishing_Legitimate_full"
    "breast-w"
    "credit-g"
    "phoneme"
    "gas-drift"
) do (
for %%S in (11 22 33 44 55) do (
    python skeleton/experiment.py --seed %%S --methods default random smbo hyperband foundation --profile "exp" --context 1000 --dataset %%D
))