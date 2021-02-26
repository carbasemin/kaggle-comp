import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

train = pd.read_csv('../train.csv')

y = train.pop('label')
X = train

# Let's make the data a bit smaller so that I don't have to wait a day.
pca = PCA(n_components=0.95)
X_ = pca.fit_transform(X)

# Bunch of base classifiers, to get a feel.
logistic = LogisticRegression(multi_class='multinomial')
ridge = RidgeClassifier(normalize=True)
nb = GaussianNB()
knn = KNeighborsClassifier(n_jobs=-1)
tree = DecisionTreeClassifier()
extra_tree = ExtraTreeClassifier()
rf = RandomForestClassifier(n_jobs=-1)

models = [logistic, ridge, nb, knn, tree, extra_tree, rf]

scores = {f'{model}': cross_val_score(model, X_, y, cv=3, n_jobs=-1).mean()
			for model in models}

# Copying the results. K-NN really likes reduced dimensions no?
# {"LogisticRegression(multi_class='multinomial')": 0.912952380952381,
#  'RidgeClassifier(normalize=True)': 0.8486666666666668,
#  'GaussianNB()': 0.8564761904761905,
#  'KNeighborsClassifier(n_jobs=-1)': 0.9665476190476191,
#  'DecisionTreeClassifier()': 0.8042857142857143,
#  'ExtraTreeClassifier()': 0.5214523809523809,
#  'RandomForestClassifier(n_jobs=-1)': 0.9376666666666668}