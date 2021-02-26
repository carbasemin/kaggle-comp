import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

y = train.pop('label')
X = train

pca = PCA(n_components=0.98)
X_ = pca.fit_transform(X)

rf = RandomForestClassifier(n_jobs=-1)

param_grid = {
'criterion': ['entropy'],
'max_depth': [8],
'max_features': ['sqrt'],
'n_estimators': [100, 150, 200]
}

clf = GridSearchCV(rf, param_grid=param_grid, cv=3)

clf.fit(X_, y)

# model = Pipeline(
# 	steps=[
# 		('pca', PCA(n_components=0.98)),
# 		('rf', RandomForestClassifier(**best_params))
# 	])

# model.fit(X, y)
# pred = model.predict(test)

# pred = pd.DataFrame(pred, index=pd.RangeIndex(1, 28001), columns=['Label'])
# pred.to_csv('../submission.csv')