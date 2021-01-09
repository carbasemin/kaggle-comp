import pandas as pd
import numpy as np

from custom_stuff import Alone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('./train.csv', index_col='PassengerId')
test = pd.read_csv('./test.csv', index_col='PassengerId')

X = data.drop(columns=['Cabin', 'Name', 'Ticket'])
y = X.pop('Survived')

test = test.drop(columns=['Cabin', 'Name', 'Ticket'])

categorical_cols = [col for col in X.columns if
                   X[col].dtype == 'object']

# First, impute the missing values with the most frequent value of the feature,
# then, onehotencode stuff. Basic.
categorical_transformer = Pipeline(
	steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(sparse=False))
		])

alone = Alone()

# So I can't get this stuff to work with Alone. Shit. I'll figure it out.
transformer = ColumnTransformer(
	transformers=[
		('num', SimpleImputer(strategy='median'), ['Pclass', 'Fare', 'Age']),
		('alone', alone, ['SibSp', 'Parch']),
		('cat', categorical_transformer, categorical_cols)
		])

model = Pipeline(
	steps=[
		('preprocessor', transformer),
		('rf', RandomForestClassifier())
	])

param_grid = { 
    'rf__n_estimators': [10, 50, 100],
    'rf__max_features': ['auto', 'sqrt', 'log2'],
    'rf__max_depth' : [4,5,6,7,8],
    'rf__criterion' :['gini', 'entropy']
}

clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=5)

clf.fit(X, y)

best_params, best_score = clf.best_params_, clf.best_score_

# To get rid of the "rf__"
keys = [key.split('__')[1] for key in best_params.keys()]
values = list(best_params.values())

best_params = {keys[i]: values[i] for i in range(len(keys))}

# The optimized model.
rf = RandomForestClassifier(**best_params)

model = Pipeline(
	steps=[
		('preprocessor', transformer),
		('rf', rf)
	])

model.fit(X, y)
y_pred = model.predict(test)

pred = pd.DataFrame(y_pred, columns=['Survived'], index=test.index)
pred.to_csv('gender_submissions.csv')