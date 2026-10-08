@REM  @echo off
@REM  for %%D in (
@REM      "breast-w"
@REM      "credit-g"
@REM      "phoneme"
@REM      "Phishing_Legitimate_full"
@REM      "gas-drift"
@REM  ) do (
@REM  for %%S in (11 22 33 44 55) do (
@REM      python skeleton/experiment.py --seed %%S --methods default random smbo hyperband foundation --profile "exp" --context 1000 --dataset %%D
@REM  ))

@echo off
for %%D in (
    "gas-drift"
) do (
for %%S in (11 22 33 44 55) do (
    python skeleton/experiment.py --seed %%S --methods foundation --profile "exp" --context 1000 --dataset %%D
))