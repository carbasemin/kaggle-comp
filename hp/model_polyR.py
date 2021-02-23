import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error

# Import the data.
train = pd.read_csv('./train.csv', index_col='Id')
test = pd.read_csv('./test.csv', index_col='Id')

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
		('encode', OneHotEncoder(handle_unknown='ignore'))
	)
)

transformer = ColumnTransformer(
	transformers=[
		('cat_imputer', cat, categorical_cols),
		('num_imputer', SimpleImputer(strategy='median'), numerical_cols)
	]
)

# Best alpha is 40, so the GridSearch says.
model = Pipeline(
	steps=[
		('preprocessor', transformer),
		('poly', PolynomialFeatures(degree=2)),
		('Lasso', Lasso(alpha=40, normalize=True, max_iter=10000))
	]
)

## Optimizing alpha.
# param_grid = {'Lasso__alpha': [20, 30, 40]}

# clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

# clf.fit(X, y)

model.fit(X, y)

# Expected kaggle score.
y_pred = model.predict(X)
kaggle = np.sqrt(mean_squared_log_error(y, y_pred))

# Submission stuff.
pred = model.predict(test)
 
pred = pd.DataFrame(pred, index=test.index, columns=['SalePrice'])
pred.to_csv('submission.csv')