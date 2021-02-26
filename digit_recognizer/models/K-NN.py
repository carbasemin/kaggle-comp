import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

y = train.pop('label')
X = train

# 3 neigbours is the best! GridSearchCV.
model = Pipeline(
	steps=[
		('pca', PCA(n_components=0.90)),
		('knn', KNeighborsClassifier(n_neighbors=3, n_jobs=-1))
	]
)

model.fit(X, y)
pred = model.predict(test)

pred = pd.DataFrame(pred, index=pd.RangeIndex(1, 28001), columns=['Label'])
pred.to_csv('../submission.csv')