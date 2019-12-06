import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import Imputer
from scipy.stats import zscore

_user_csv = os.path.join(os.path.dirname(__file__), 'data', 'Users.csv')
_book_csv = os.path.join(os.path.dirname(__file__), 'data', 'Books.csv')
_ue_csv = os.path.join(os.path.dirname(__file__), 'data', 'UserEvents.csv')

_impression_to_rating = {
	'dislike': 0,
	'view': 0.2,
	'interact': 0.4,
	'like': 0.6,
	'add to cart': 0.8,
	'checkout': 1
}

_features = {'user:location': 'categorical', 'use:age': 'numerical',
'book:bookName': 'categorical', 'book:author': 'categorical', 
'book:publisher': 'categorical', 'book:urlId': 'categorical',
'book:yearOfPublication': 'numerical'
}


def load_dataframes(user_csv=_user_csv, book_csv=_book_csv, ue_csv=_ue_csv):
	"""
	Parameters:
		user_csv - string. Path to Users.csv
		book_csv - string. Path to Books.csv
		ue_csv - string. Path to UserEvents.csv

	Returns:
		user_df - pd.Dataframes
		book_df - pd.Dataframes
		ue_df - pd.Dataframes

	"""
	user_df = _load_csv(user_csv)
	book_df = _load_csv(book_csv)
	ue_df = _load_csv(ue_csv)

	for index in ['location']:
		user_df = _clean_special_chars(user_df, index)
	for index in ['bookISBN', 'author', 'publisher', 'bookName']:
		book_df = _clean_special_chars(book_df, index)
	for index in ['bookId']:
		ue_df = _clean_special_chars(ue_df, index)

	user_df = _impute_and_normalize(user_df, 'age')
	book_df = _impute_and_normalize(book_df, 'yearOfPublication', missing_values=0)
	book_df['urlId'] = book_df['urlId'].astype(str)

	filtered_ue = ue_df[(ue_df['user'].isin(user_df['user'])) & (ue_df['bookId'].isin(book_df['bookISBN']))]

	return (user_df, book_df, filtered_ue)


def _load_csv(csv, sort='', encoding='latin-1'):
	"""
	Parameter: 
		csv - str. '/path/to/csv'
		sort - str. index to be sorted
		encoding - str. encoding of the csv file
	Returns:
		df - pandas.DataFrame object
	"""
	df = pd.read_csv(csv, encoding=encoding)
	return df.sort_values(by=[sort]) if sort != '' else df

def _clean_special_chars(df, index, pat='[^a-zA-Z.\d\s]'):
	"""
	Parameter:
		df - pandas.DataFrames. 
		index - string. Index of the df to be  cleaned.
		pat - string. Regex expression.
	Returns:
		df - pandas.DataFrames
	Notes:
		Remove special characters except for space in a column of pd.DataFrame with index based on regex expression.
	"""
	df[index] = df[index].astype(str).str.replace(pat, '')
	return df

def _impute_and_normalize(df, index, missing_values='NaN'):
	"""
	Parameter:
		df - pandas.DataFrames. 
		index - string. Index of the df to be  cleaned.
		missing_values - string. 
	Returns:
		df - pandas.DataFrames
	Notes:
		Fills in the missing values with the average of the dataframe.
	"""

	if missing_values == 'NaN':
		df[index] = df[index].fillna(df[index].mean())
		df[index] = pd.Series(zscore(df[index].values))
	else:
		df[index] = df[index].fillna(missing_values)
		_index = np.reshape(df[index].values, (len(df[index]), 1))

		imputer = Imputer(missing_values=missing_values, strategy='mean')
		_imputed = imputer.fit_transform(_index)
		_zscores = zscore(np.reshape(_imputed, (len(_imputed),))).tolist()
		df[index] = pd.Series(_zscores)

	return df
