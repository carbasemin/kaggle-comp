import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('./train.csv')

# y = train.pop('SalePrice')

corr = train.corr()

# # Heatmap, the best way to make sense of the initial soup.
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corr, vmax=.8, square=True)
# f.savefig('corr.png')
# # Investigate multicolinearity; the variables that say the same thing, or almost.

k = 10
cols = corr.nlargest(k, 'SalePrice')['SalePrice']
cols.drop(['GarageCars', 'TotalBsmtSF'], inplace=True)

# Then scatter plots to see the relationships between cool variables? Why not?

# Maybe normalise the stuff? Yep, it helps? Maybe.

#--- %1

# Normalize