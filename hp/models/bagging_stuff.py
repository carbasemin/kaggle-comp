import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score

# Import the data.
train = pd.read_csv('../train.csv', index_col='Id')
test = pd.read_csv('../test.csv', index_col='Id')

# The features that'll create too much noise in the imputting phase.
mask = train.isnull().sum() > 100
noisy = train.columns[mask]
train.drop(noisy, axis=1, inplace=True)
test.drop(noisy, axis=1, inplace=True)

X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
test = test[X.columns]

# Getting the numerical and the categorical variables for polynomial expansion.
categorical_cols = [col for col in X.columns if
					X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if
					col not in categorical_cols]

cat = Pipeline(
	steps=(
		('impute', SimpleImputer(strategy='most_frequent')),
		('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
	)
)

transformer = ColumnTransformer(
	transformers=[
		('cat_imputer', cat, categorical_cols),
		('num_imputer', SimpleImputer(strategy='median'), numerical_cols)
	]
)

# Best alpha is 40, so the GridSearch says.
preprocessor = Pipeline(
	steps=[
		('preprocessor', transformer),
		('poly', PolynomialFeatures(degree=2)),
		('scaler', MinMaxScaler())
	]
)

X_ = preprocessor.fit_transform(X)
test_ = preprocessor.transform(test)
y_ = np.log1p(y)

lasso = Lasso(alpha=40, max_iter=10000)
bag_reg = BaggingRegressor(lasso, n_estimators=50, bootstrap=True)

score = cross_val_score(bag_reg, X_, y_, cv=5, scoring='neg_mean_squared_error')

# Submission stuff.
# bag_reg.fit(X_, y)
# pred = bag_reg.predict(test_)
 
# pred = pd.DataFrame(pred, index=test.index, columns=['SalePrice'])
# pred.to_csv('../submission.csv')