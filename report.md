# Report 

## TODO
1. data exploration
2. metric selection: consider datasets, whether large or small values for metrics are desired
3. search space definition: find all relevant hyperparameters. Determine ranges, whether to perform transformations
4. formalize compute metric: number of full sized forests trained (use the number of trees and max tree size of a given forest)
5. Make explicit all relevant optimizer settings and justify: for Random, what probability distribution, for SMBO, what surrogate model, acquisition function, etc., for Hyperband what hyperparameters.
6. Double check what will need to be logged to answer our 3 research questions
7. Formalize tab fm experiment

### Introduction

##### Research questions
1. How do the optimisers compare with the untuned forest across the five datasets?
2. How efficiently do they find good configurations?
3. How does the foundation-model approach compare with all forest methods in predictive
performance and computational effort?


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

- metric: balanced accuracy, can be used to evaluate on validation set. For imbalanced datasets it is a useful metric. Otherwise ROC-AUC or F1.



##### Search space 
do we use parameter transformations like in (hutter2011sequential)? Useful for large min and max of a hyperparam range. 
- define each hyperparameter: range, scale, and constraints
- class_weight: it might be worth setting this prior to search, as it is generally known to be good to use class weights and the value-add of searching for this is minimal (esp. in random search). Otherwise, only between {"None", “balanced”, “balanced_subsample”}.
- min_samples_split: int, in {0, ..., 10} number of samps required to split an internal node
- min_samples_leaf: int in {1, ?}

##### Compute metric
- running time
- number of trees trained (unless we determine a fixed budget), to make it standard across all methods, number of trees divided by the max size of a full-sized tree. Final refit on val+train is not included as a cost.
  

##### Optimizer settings
**common max tree count **

- Untuned
Default settings. 

- Random Search
Monte-carlo sampling 

- SMBO
BoTorch. Surrogate model: GP. 

- Hyperband
Are trees reused (warmstart), this should be taken into account in number of trees trained. 

##### Seeds 
- 5 random seeds
the dataset strata  (train +val) can vary as well

give each method the same number of validation evaluations (number of trees trained). Compute the number for hyperband, then give the same budget to SMBO and Random Search

##### Tabular foundation model
- in-context learning
- TabPFN3.5 (latest)

##### Logging
- all evaluations: validation and train metrics. This shows the optimizer process across all optimizers. Also the final model performance which is used for the refitting experiment. 
- configurations (search space vector for a given evaluated tree, for hyperband this should also be logged for intermediate trees)



##### Reproducibility

##### 





### Results

### Conclusion

### References

@inproceedings{hutter2011sequential,
  title={Sequential model-based optimization for general algorithm configuration},
  author={Hutter, Frank and Hoos, Holger H and Leyton-Brown, Kevin},
  booktitle={International conference on learning and intelligent optimization},
  pages={507--523},
  year={2011},
  organization={Springer}
}
