# %1

Which variables are categorical, which variables are quantative.
Normalize the target, it's not normal, that's why.
Sperman correlation can pick up relationships that are non-linear. How the fuck?
Heatmaps, of course.
tSNE, PCA, StandardScaler, KMeans clustering.
He's doing some crazy shit man...

# %0.3

Most ML models don't do well with non-normally distributed data. OK. log(1+x) transform to the SalePrice then.
Missing values and stuff.
Find relevant features.
Normalize the features also.
Create a tone of features I believe.
Encode categorical features.
Train a bunch of models and create an ensemble with StackingCVRegressor.
Then blend all the models.

# top4

If some variable is categorical but it looks numeric, then, use apply(str) on them or something.
LabelEncoder. What did this do?
Box Cox transformation to tame the skewed features.
Getting dummies.
If some model is sensitive to outliers, use RobustScaler(), make a pipeline, be a neighbour.