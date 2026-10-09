# AutoML

## Instructions
These instructions assume UV package manager. Either remove `uv run` from .bat and .sh files, or install uv. 

1. in root folder
```bash
pip install -r skeleton/requirements.txt
```
2. to be able to run tabular foundation model: first make an account for prior labs.
Add your API key in a local .env file and add to .gitignore file. DO NOT SHARE YOUR KEY.
This enables you to use the client, otherwise, use tabpfn locally.

3. run the experiments
```bash
experiment.bat
```

4. produce plots from resulting data
```bash
python skeleton/plotting.py
```
