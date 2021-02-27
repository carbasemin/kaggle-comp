import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

y = train.pop('label')
X = train

scaler = StandardScaler()

X_ = scaler.fit_transform(X)

svc = LinearSVC(loss='hinge')

score = cross_val_score(svc, X_, y, cv=3) # ~0.90 Not good.