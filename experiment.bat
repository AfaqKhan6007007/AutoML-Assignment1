@echo off
for %%S in (11 22 33 44 55 66 77 88 99 100) do (
    python skeleton/experiment.py --seed %%S --methods default random smbo hyperband foundation --profile "exp" --context 1000
)