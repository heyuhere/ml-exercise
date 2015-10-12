# Machine Learning exercise

### How to run?
Simply run `bash run.sh <training data file> <testing data file>` or `./run.sh <training data file> <testing data file>` from command line. It will produce `train.csv` and `test.csv` files with `\x00` characters removed, and execute the python `model.py` script. The result (CONTROLN, TARGET_B, and TARGET_D for test data) will be stored in `output.csv` file.


### Why python?

Strenghts
- many data analysis, ML tools (scikit-learn, pandas, etc)
- concise expression (10 lines of python equivalent to 100 lines of Java)
- integration with high-performance libraries for runtime performance(numpy, scipy, numba, cpython, etc)
- graphics support (matplotlib, seaborn, D3)
- general programming language (as opposed to R, matlab, octave, SAS)
- vibrant data analytics community that keeps developing tools
Considerations
- relatively bigger memory footprint than some other languages such as Java, C++
- slow execution speed when written purely in python



### High-level modeling

There are 2 target variables (`TARGET_B`, `TARGET_D`) where `target_b` is a binary variable and `target_d` is a continous-value variable. Using these facts, I split the problem into two:
 - Is the person likely to donate? (`target_b`: classification)
 - If so, how much? (`target_d`: regression)



### Data analysis process

- load data
- preprocess data
  - clean up data
  - drop irrelevant features
  - normalize features
- train model with training data
- predict with testing data



#### Preprocessing

##### Data cleanup

Some data has missing values. In order to avoid models complaining about missing data, 0s were filled in place of N/A fields.

##### Dimensionality reduction

When there are a lot of features to analyze, it usually helps to pick subset of them because:
- it requires less runtime, and hence, quicker production performance and model training
- it saves storage space to keep relevant features

A few different ways to cope with high-dimension data set:
- Feature selection: step-wise approach, correlation analysis
- Feature extraction: PCA/Kernel PCA

Of the 2 different approaches, I chose `Feature selection` as data is already given in this case and I just needed to choose which ones to use for model training. There are also a few techniques for feature selection:
- Filter (correlation or some other metric) analysis
  - Pick the lowest/highest score by the metric and select or drop the feature
  - It could be also used to remove multi-collinearity to get rid of features that provide overlapping information
- Manifold learning
  - Non-linear dimensionality reduction technique.
  - Not used here because of its [time complexity](http://scikit-learn.org/stable/modules/manifold.html#id7)

Now, let's get into details of each technique:

###### PCA/Kernel PCA

PCA (Principal Component Analysis) transforms data orthogonally to yield linearly uncorrelated values, namely `principal components`. Kernel PCA adds `kernel trick` to PCA in order to find out non-linearity in data.

I tried Kernel PCA with RBF kernel, however, it gave me `segmentation fault: 11`. Googling gave me some possible patches but unfortunately, none of them seemed to work.

###### Filter analysis

Correlation analysis is used to kick out cause of multi-collinearity and also keep linearly relevant features. I decided to provide `L1 penalty` to models as it orchestrates the same task within model building process.



##### Data normalization

When it comes to model training, iterative algorithms are used to find the optimal parameters. These iterative algorithms depend on numerical analysis to find out the next parameters to try out. Here, if the range of data is not consistent across different features, the parameter finding process might not converge. In order to ensure stable and gradual convergence of model parameter search, data need to be normalized.

There are 2 categories of data: numerical and categorical

###### Numerical feature

Numerical values can be scaled to remove mean and to have unit variance of 1. This way, all features provide the same kind of movement in optimal parameter search.

###### Categorical feature

Categorical features are mapped to integers as models can understand integers.





### Statistical models

I have tried a few different models, both linear and non-linear. And here is brief (upside) comparison between the two groups.
linear | non-linear
------ | ----------
- fast | - higher accuracy
- dimensionality reduction bundled in model construction with L1 penalty | - good trade-off of runtime speed and prediction accuracy
- lower memory consumption | 

More in-depth analysis on the experimented models follow below:


#### Support Vector Machine (SVM)

- Easy to switch back and forth between linear/non-linear models
- Easy to integrate kernel trick for non-linearity
- Works well for high dimensional data
- Slow training and prediction (compared to linear models)


#### SGD (stochastic gradient descent)

- Super fast speed
- Online learning (no need to have all data set ready when training model)
- No support for linearity (could apply kernel trick on data set before training)
- [Vowpal Wabbit]() implements SGD classification with multi-node scalability


### Notes

#### Challenges
- Environment issue: While trying to apply non-linear techniques, I kept getting `segmentation fault: 11`.
- Resources issue: When the above error was not thrown, memory was maxed out even though all the 64-bit data was converted to 32-bit to save some space. In other words, even cutting data size by half did not seem to help as I kept reaching the memory limit. And I suspect it might be because of the nature of Python. It consumes more memory than other languages. I did not think I would run into this kind of issue with 100K by 500 data set and thus chose python, but since it does, I might look at other platforms next time.

#### Assumption
#6 in `IMPORTANT ADDITIONAL NOTES` in the instructions, it states that
```
For each record, there are two target/dependent variables
(field names: TARGET_B and TARGET_D). TARGET_B is a binary variable
indicating whether or not the record responded to the promotion of
interest ("97NK" mailing) while TARGET_D contains the donation amount
(dollar) and is only observed for those that responded to the
promotion. **Neither of these should be used to train the model**.
```
But since this is a classification, regression problem, dependent variable (Y) is required. My modeling is based on the assumption that the two target variables can be used to train the model.

#### Other directions of analysis

After running into environment issues and limitations from resources, I came to think about other approaches that I could take to get better performance and stability. This still may not work as it is just a theory, but I think there is a good chance it still will as it is designed to take less resources.
- Kernel trick with RBF: The variables seem to have non-linear relationships that can only be exploited by proper non-linear models. RBF can fit to any shape of data distribution and so is a good option to explore.
- step-wise feature selection: At every round, select one variable with the highest score. Variables will be added until the model meets stop criteria. (# of variables, % of variance explained, accuracy, etc) This may result in longer execution time, but will take less resources.
- SGD or SVM: Since RBF kernel would be already applied above, the model would not need to be non-linear. SGD is a linear model that provides super fast online learning. It is known to scale well to a cluster of machines. The reason SVM might be still worth a shot is its adaptibility to high-dimensional space.
