import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict

data = pd.read_csv('train.csv', index_col='Id')
test = pd.read_csv('test.csv', index_col='Id')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Categorical and numerical features.
categorical_cols = [col for col in X.columns if 
				X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if 
				X[col].dtype != 'object']

# First, impute the missing values with the most frequent value of the feature,
# then, onehotencode stuff. Basic.
categorical_transformer = Pipeline(
	steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(handle_unknown='ignore'))
		])

transformer = ColumnTransformer(
	transformers=[
		('num', SimpleImputer(strategy='median'), numerical_cols),
		('cat', categorical_transformer, categorical_cols)
		])

model = Pipeline(
	steps=[
		('preprocessor', transformer),
		('rf', RandomForestRegressor(n_jobs=-1))
	])

model.fit(X, y)

pred = model.predict(test)

pred = pd.DataFrame(pred, index=test.index, columns=['SalePrice'])
pred.to_csv('submission.csv')