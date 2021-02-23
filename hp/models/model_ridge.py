import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('../train.csv', index_col='Id')
test = pd.read_csv('../test.csv', index_col='Id')

X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Since we're using all the features, there'll be some cleaning.
# So, let's repeat the usual steps.
categorical_cols = [col for col in X.columns if
					X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if
					col not in categorical_cols]

# Scaling categorical and numerical columns seperately, not sure about this.
# Not sure how the StandartScaler works either. I will look into it.
categorical_transformer = Pipeline(
	steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore')),
		('scaler', StandardScaler())
	]
)

numerical_transformer = Pipeline(
	steps=[
		('imputer', SimpleImputer(strategy='median')),
		('scaler', StandardScaler())
	]
)

transformer = ColumnTransformer(
	transformers=[
		('num', numerical_transformer, numerical_cols),
		('cat', categorical_transformer, categorical_cols),
	]
)

# Best alpha seems to be 750. Look below!
model = Pipeline(
	steps=[
		('preprocessor', transformer),
		('ridge', Ridge(750))
	]
)

## Optimization.
# param_grid = {'ridge__alpha': [1, 10, 100, 250, 500, 750, 1000, 2500]}

# clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=5)

# clf.fit(X, y)

model.fit(X, y)

pred = model.predict(test)

pred = pd.DataFrame(pred, index=test.index, columns=['SalePrice'])
pred.to_csv('../submission.csv')