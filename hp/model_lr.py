import pandas as pd

from sklearn.linear_model import LinearRegression

# Import the data.
train = pd.read_csv('./train.csv', index_col='Id')
test = pd.read_csv('./test.csv', index_col='Id')

# Using the variable with the highest correlation.
X = train[['OverallQual'], ]
y = train['SalePrice']
test = test[['OverallQual']]

model = LinearRegression()

model.fit(X, y)
pred = model.predict(test)

pred = pd.DataFrame(pred, index=test.index, columns=['SalePrice'])
pred.to_csv('submission.csv')