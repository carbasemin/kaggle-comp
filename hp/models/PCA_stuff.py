import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_log_error

# Import the data.
train = pd.read_csv('../train.csv', index_col='Id')
test = pd.read_csv('../test.csv', index_col='Id')

# The features that'll create too much noise in the imputting phase.
mask = train.isnull().sum() > 100
noisy = train.columns[mask]
train.drop(noisy, axis=1, inplace=True)
test.drop(noisy, axis=1, inplace=True)

# Seperating features, label and preparing the test.
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
test = test[X.columns]

# Getting the numerical and the categorical variables for polynomial expansion.
categorical_cols = [col for col in X.columns if
					X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if
					col not in categorical_cols]

# Categorical preprocessor.
cat = Pipeline(
	steps=(
		('impute', SimpleImputer(strategy='most_frequent')),
		('encode', OneHotEncoder(sparse=False ,handle_unknown='ignore'))
	)
)

transformer = ColumnTransformer(
	transformers=[
		('cat_imputer', cat, categorical_cols),
		('num_imputer', SimpleImputer(strategy='median'), numerical_cols)
	]
)

# From below.
best_params={
'n_estimators': 600,
'min_samples_split': 5,
'min_samples_leaf': 2,
'max_features': 'sqrt',
'max_depth': 40,
'bootstrap': True}

model = Pipeline(steps=[
	('preprocess', transformer),
	('pca', PCA(n_components=4)),
	('RF', RandomForestRegressor(**best_params, n_jobs=-1))
])

# param_grid = {
# 	'RF__bootstrap': [True, False],
# 	'RF__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
# 	'RF__max_features': ['auto', 'sqrt'],
# 	'RF__min_samples_leaf': [1, 2, 4],
# 	'RF__min_samples_split': [2, 5, 10],
# 	'RF__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# }

# clf = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
# 						n_jobs=-1, cv=3, n_iter=50,
# 						scoring='neg_mean_squared_log_error')

# clf.fit(X, y)

model.fit(X, y)

pred = model.predict(test)

pred = pd.DataFrame(pred, index=test.index, columns=['SalePrice'])
pred.to_csv('../submission.csv')