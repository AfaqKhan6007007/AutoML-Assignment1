# Report 

### Introduction

### Methods

##### Data protocol

- data split
stratified holdout (keeps the data proportions across the strata) 
define train + val/ test split, during random seeds the train+val splits can be changed,
however, the test split remains untouched across all optimizer runs and seeds to avoid test
data leakage.

- preprocessing and final refitting
Some of the datasets will require some preprocessing (e.g. one hot encoding, empty entries)
During final refitting/ training of the optimized models, train on the entire train+val combined data
then test on the held out test set

  
##### Metric and objective function

- fair comparison
For fair comparison it is important to consider that multiclass and binary classification problems have different
metrics. Find a metric which is consistent across the datasets.

Also consider metrics which are more robust to class imbalances that are present in the data. Report the effect of class imbalance on perceived performance.

##### Search space 

- define each hyperparameter: range, scale, and constraints
- common max tree count

##### Compute metric
- running time
- number of trees trained (unless we determine a fixed budget)

##### Optimizer settings
- Random Search

- SMBO

- Hyperband

##### Seeds 
- 5 or 10 random seeds
the dataset strata  (train +val) can vary as well

##### Tabular foundation model
- in-context learning
- TabPFN3.5 (latest)

##### Logging

##### Reproducibility

##### 





### Results

### Conclusion

### References
