# AutoML

## To-do:
1. run experiments for all seeds, with bash command (see below).
2. write plotting functions for relevant results (e.g. SMBO, Random search, balanced accuracy curves plotted against n_trials. For hyperband plot all evaluations as accuracy against resource: only promoted forests continue. For all consider using mean/std across runs.)
3. reconsider number of seeds vs number of trials, might yield more interesting curves. Do a trial SMBO / Random search run first to decide.
4. summarize findings in report
5. add refs in report
6. write introduction
7. keep report as concise as possible (use tables where possible to avoid excessive explanations)

## Instructions
1. in root folder
```bash
pip install -r requirements.txt
```
2. to be able to run tabular foundation model: first make an account for prior labs.
Add your API key in a local .env file and add to .gitignore file. DO NOT SHARE YOUR KEY.

3. run bash commands with args to run experiments

```python
seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
```

```bash
python skeleton/experiment.py --seed 11 --methods default random smbo hyperband foundation --profile "exp"
```

