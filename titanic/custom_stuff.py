from sklearn.base import BaseEstimator, TransformerMixin

# A custom transformer that creates a is_alone column.
# This could be generalized to whatever column ya' know.
# Let's just do our job though.

class Alone(BaseEstimator, TransformerMixin):
	'''Adds the is_alone column to the dataframe, which, as the name suggest,
	shows if the pessenger is alone or not; this stuff also drops SibSp and
	Parch features since we use them to compute the is_alone feature.'''

	def fit(self, X, y=None):
		return self
	
	def transform(self, X, y=None):
		# Calculate the family size.
		family_size = X['SibSp'] + X['Parch']

		# Create the is_alone column and assume everyone is alone.
		X['is_alone'] = 0
		# If the family size is not 0, then the person is not alone. Duh. 
		X.loc[family_size != 0, 'is_alone'] = 1

		# We'll get rid of SibSp and Parch now.
		X.drop(['SibSp', 'Parch'], axis=1, inplace=True)

		return X