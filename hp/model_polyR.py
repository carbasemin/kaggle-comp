import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Import the data.
train = pd.read_csv('./train.csv', index_col='Id')
test = pd.read_csv('./test.csv', index_col='Id')

# Create a mask for features that are "highly corellated" with SalePrice. 
mask = train.corr()['SalePrice'].values > 0.5
# Get the feature name for feature engineering.
relevant = train.corr()['SalePrice'][mask].index.tolist()

# GarageCars and GarageArea - TotalBsmtSF and 1stFlrSF - GrLivArea and TotRmsAbvGrd  
# are highly corraleted. I'll go with the former of those.
relevant = [feature for feature in relevant if
			feature not in ('GarageArea', '1stFlrSF', 'TotRmsAbvGrd')]

X = train[relevant].drop('SalePrice', axis=1)
y = train[relevant]['SalePrice']
test = test[X.columns]

# There's no null values in the training set, but there is in the test.
# TotalBsmtSF is numerical and GarageCars is categorical, so, different imputers.

transformer = ColumnTransformer(
	transformers=[
		('cat_imputer', SimpleImputer(strategy='most_frequent'), ['TotalBsmtSF']),
		('num_imputer', SimpleImputer(strategy='median'), ['GarageCars'])
	]
)

model = Pipeline(
	steps=[
		('preprocessor', transformer),
		('poly', PolynomialFeatures(degree=2)),
		('regressor', LinearRegression())
	]
)

model.fit(X, y)
pred = model.predict(test)
 
pred = pd.DataFrame(pred, index=test.index, columns=['SalePrice'])
pred.to_csv('submission.csv')