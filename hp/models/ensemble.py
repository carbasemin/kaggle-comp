import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor

train = pd.read_csv('../train.csv', index_col='Id')
test = pd.read_csv('../test.csv', index_col='Id')

X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

## ----- Preprocessing for Lasso.
# Getting the numerical and the categorical variables for polynomial expansion.
categorical_cols = [col for col in X.columns if
					X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if
					col not in categorical_cols]

# What's numerical and what's categorical?
for col in numerical_cols:
	X[col].fillna(X[col].median(), inplace=True)
	
for col in categorical_cols:
	X[col].fillna(X[col].mode()[0], inplace=True)

# Getting the relevant categorical features.
dummies = pd.get_dummies(X[categorical_cols])

# Got the alpha from GridSearchCV and then got rid of the steps to simplify code.
lasso_c = Lasso(alpha=25, normalize=True)

lasso_c.fit(dummies, y)

mask = (lasso_c.coef_ > 1)

# Turning them into a pd.Series to use it as a mask.
relevant_cat_cols = dummies.columns[mask]
# Since we used pd.get_dummies, the relevant columns, i.e., columns with weights
# greater than the rather arbitrary value of 1, will have names like
# 'LotConfig_Corner', 'LotConfig_CulDSac', 
# 'Neighborhood_Crawfor', 'Neighborhood_NPkVill', and so on.
relevant_cat_cols = set([col.split('_')[0] for col in relevant_cat_cols])

# Getting the relevant numerical features
nums = X[numerical_cols]

# Got the alpha from GridSearchCV like before.
lasso_n = Lasso(alpha=15, normalize=True)

lasso_n.fit(nums, y)

mask = (lasso_n.coef_ > 1)

# Turning the list into a pd.Series to index them with the mask.
num = pd.Series(numerical_cols)
# Getting them in a set because I want to merge them with the categoricals. 
relevant_num_cols = set(num[mask])

# Getting the union of them in a list, because you can't 
relevant_cols = relevant_num_cols.union(relevant_cat_cols)

X_rel = X[relevant_cols]
## ----- Preprocessing for Lasso.

# I manually filled the missing values above but the test set needs the same step and
# using the values of the test set to fill its missing values is not good practice.
categorical_transformer = Pipeline(
	steps=[
		('cat_imputer', SimpleImputer(strategy='most_frequent')),
		('ohc', OneHotEncoder(handle_unknown='ignore'))
	]
)

preprocessor = ColumnTransformer(
	transformers=[
		('cat', categorical_transformer, list(relevant_cat_cols)),
		('num', SimpleImputer(strategy='median'), list(relevant_num_cols))
	]
)

X_lasso = preprocessor.fit_transform(X_rel)
test_rel = test[relevant_cols]
test_lasso = preprocessor.transform(test_rel)

X_gbr = preprocessor.fit_transform(X)
test_gbr = preprocessor.transform(test)

# Poly-Lasso.
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_lasso)
test_poly = poly.transform(test_lasso)

lasso = Lasso(alpha=30, normalize=True)

lasso.fit(X_poly, y)
lasso_pred = lasso.predict(test_poly)

# Gradient boosting.
best_params = {
'learning_rate': 0.1,
'max_depth': 4,
'max_features': 0.3,
'min_samples_leaf': 9,
'n_estimators': 200
}

gbr =  GradientBoostingRegressor(**best_params)

gbr.fit(X_gbr, y)
gbr_pred = gbr.predict(test_gbr)

# Submission stuff.
pred = 0.8*lasso_pred + 0.2*gbr_pred
 
pred = pd.DataFrame(pred, index=test.index, columns=['SalePrice'])
pred.to_csv('../submission.csv')