import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

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

model = Pipeline(steps=[
	('preprocess', transformer),
	('pca', PCA(n_components=4)),
	('RF', RandomForestRegressor(n_jobs=-1))
])

model.fit(X, y)

pred = model.predict(test)

pred = pd.DataFrame(pred, index=test.index, columns=['SalePrice'])
pred.to_csv('../submission.csv')