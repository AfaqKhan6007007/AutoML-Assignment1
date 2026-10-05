# Report 

### Introduction

### Methods

##### Data protocol

- data split
stratified holdout
define train + val/ test split, during random seeds the train+val splits can be changed,
however, the test split remains untouched across all optimizer runs and seeds to avoid test
data leakage.

- preprocessing and final refitting
Some of the datasets will require some preprocessing (e.g. one hot encoding, empty entries)
During final refitting/ training of the optimized models, train on the entire train+val combined data
then test on the held out test set

- data leakage & fair comparison
For fair comparison it is important to consider that multiclass and binary classification problems have different
metrics. Find a metric which is consistent across the datasets. 
  
##### 



### Results

### Conclusion

### References
