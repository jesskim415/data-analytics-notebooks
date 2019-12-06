import pandas as pd
import numpy as np
import os
from data_handler import load_dataframes, _impression_to_rating, _features
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import get_tmpfile



class Dataset:
	def __init__(self):
		(self.user_df, self.book_df, self.ue_df) = load_dataframes()
		# Indexes of the filtered UserEvents dataframe from original UserEvents dataframe
		self.ue_index = self.ue_df.index.values

		self.n_users = self.user_df.shape[0]
		self.n_books = self.book_df.shape[0]
		self.n_ue = self.ue_df.shape[0]

		self.userid_to_index = {s: i for i, s in enumerate(self.user_df['user'].tolist())}
		self.bookid_to_index = {s: i for i, s in enumerate(self.book_df['bookISBN'].tolist())}

		self.ratings = np.empty((self.n_users, self.n_books))
		self.ratings[:] = np.nan

		self.known_index = None
		print('Initializing dataset .. ')
		print('There are total of %d users, %d books, and %d relevant user-events.' 
			% (self.n_users, self.n_books, self.n_ue))

		################
		## This part if for collaborative filtering.
		
		self.known_users = self.ue_df['user'].unique()
		self.known_user_to_index = {s: i for i, s in enumerate(self.known_users)}
		self.index_to_known_user = {i: s for i, s in enumerate(self.known_users)}

		self.known_books = self.ue_df['bookId'].unique()
		self.known_book_to_index = {s: i for i, s in enumerate(self.known_books)}
		self.index_to_known_book = {i: s for i, s in enumerate(self.known_books)}

		self.known_ratings = np.empty((len(self.known_users), len(self.known_books)))
		self.known_ratings[:] = np.nan

		self.mu_users = None
		self.mu_books = None

		##################

		print('There are total of %d known users and %d known books engaged in user-events.' % (len(self.known_users), len(self.known_books)))
		
		self.user_cattovec = Category2Vector()
		self.book_cattovec = Category2Vector()

		self.user_features = None
		self.user_dim = 0

		self.book_features = None
		self.book_dim = 0

		self.ue_features = None


	def get_ratings_from_user_events(self):
		"""
		Loads ratings from UserEvents.
		'Impression' is transformed to numerical value 'rating' via _impression_to_rating dict in data_handler.py
		Numbers in 'rating' is empirical and is subject to change.

		"""
		for i in range(self.n_ue):
			ue = self.ue_df.loc[self.ue_index[i]]
			user_index = self.userid_to_index[ue['user']]
			book_index = self.bookid_to_index[ue['bookId']]

			self.ratings[user_index, book_index] = _impression_to_rating[ue['impression']]

			known_user_index = self.known_user_to_index[ue['user']]
			known_book_index = self.known_book_to_index[ue['bookId']]

			self.known_ratings[known_user_index, known_book_index] = _impression_to_rating[ue['impression']]

		self.mu_users = np.nanmean(self.known_ratings, axis=0)
		self.mu_books = np.nanmean(self.known_ratings, axis=1)

		self.known_index = np.argwhere(np.invert(np.isnan(self.ratings))).tolist()

		print('Retrieving rating .. ')
		return self.ratings

	def get_user_features(self, c2v_file, c2v_size=8):
		"""
		Parameter:
			c2v_file - filepath to user category-to-vec model.
			c2v_size - size of the vector of category-to-vec. Default is 8.

		Returns:
			user_features - matrix of user features by (n_users * user_dim)
			user_c2v - Category2Vector object
			user_dim - int. size of user features vector
			
		Note:
			For user features, both 'location' and 'age' are used as default.
		"""
		print('Calculating user features .. ')

		self.user_dim = c2v_size + 1
		self.user_features = np.zeros((self.n_users, self.user_dim))
		if os.path.isfile(c2v_file):
			self.user_cattovec.load_model(c2v_file)
		else:
			print('Could not locate c2v model. Training model .. ')

			self.user_cattovec.train_categories(size=c2v_size,
												df_index_list=[[self.user_df, ['location']]],
												save_model=os.path.abspath(c2v_file))

		print('[USER] Categorical features: location')
		print('[USER] Numerical features: age')

		for i in range(self.n_users):
			user = self.user_df.loc[i]

			location = self.user_cattovec.get_vector_from_category(user['location'].lower(), size=c2v_size)
			age = [[user['age']]]

			self.user_features[i,:] = np.c_[location, age]
		
		print('Done calculating user features.')

		return self.user_features, self.user_cattovec, self.user_dim


	def get_book_features(self, features, c2v_file, c2v_size=8):
		"""
		Parameter:
			features: List of book features to be used.
			c2v_file: filepath to c2v model.

		Returns:
			book_features - matrix of book features by (n_users * book_dim)
			book_c2v - Category2Vector object
			book_dim - int. size of book features vector

		Notes:
			Because of data sparsity, categorical features are transformed to vectors by Word2Vec.

		"""
		print('Calculating all book features..')

		counter = {'categorical': 0, 'numerical': 0}
		categorical = []
		numerical = []
		for feature in features:
			counter[_features['book:' + feature]] += 1
			eval(_features['book:' + feature]).append(feature)

		print('[BOOK] Categorical features: {}'.format(', '.join(categorical)))
		print('[BOOK] Numerical features: {}'.format(', '.join(numerical)))

		self.book_dim = counter['categorical'] * c2v_size + counter['numerical']
		self.book_features = np.zeros((self.n_books, self.book_dim))

		if os.path.isfile(c2v_file):
			self.book_cattovec.load_model(c2v_file)
		else:
			self.book_cattovec.train_categories(size=c2v_size, 
												df_index_list=[[self.book_df, features]],
												save_model=os.path.abspath(c2v_file))

		for i in range(self.n_books):
			book = self.book_df.loc[i]
			feature = [[book[numerical[0]]]]

			for cat in categorical:
				feature = np.c_[feature, self.book_cattovec.get_vector_from_category(book[cat].lower(), size=c2v_size)]

			self.book_features[i,:] = feature
		
		print('Done calculating book features.')

		return self.book_features, self.book_cattovec, self.book_dim


	def get_ue_features(self):
		self.ue_features = np.zeros((len(self.known_index), self.user_features.shape[1]+self.book_features.shape[1]))
		for i in range(len(self.known_index)):
			ue_feature = np.r_[self.user_features[self.known_index[i][0],:], self.book_features[self.known_index[i][1],:]]
			self.ue_features[i,:] = np.reshape(ue_feature, (1, ue_feature.shape[0]))

		return self.ue_features



class Category2Vector:
	def __init__(self, min_count=1, window=4):
		self.word_list = []
		self.model = None
		self.min_count = min_count

	def load_model(self, filepath):
		"""
		Parameter:
			filepath - string. Path to the cat2vec model. Is it does not exist, the model will be written here.
		Returns:
			Word2Vec model
		"""
		self.model = Word2Vec.load(filepath)

		return self.model

	def train_categories(self, size, df_index_list, save_model=''):
		self.model = Word2Vec(size=size, min_count=self.min_count)
		"""
		Parameter:
			size - int. Size of the vector
			df_index_list - list. df_index_list[i][0] has the dataframe and df_index_list[i][1] has list of indexes to be trained,
			save_model - string. Path where the model is saved.
		Returns:
			Word2Vec model
		"""

		category_list = []
		for df_index in df_index_list:
			category_list.extend([str(i).lower() for sub in df_index[0][df_index[1]].values.tolist() for i in sub])
		self.word_list = [p.split(' ') for p in category_list]

		self.model.build_vocab(self.word_list)	
		self.model.train(self.word_list,  total_examples=self.model.corpus_count, epochs=self.model.iter)

		if save_model:
			fname = get_tmpfile(save_model)
			self.model.save(fname)

		return self.model

	def get_vector_from_category(self, cat, size):
		"""
		Parameter:
			cat: string. Category to be vectorized.
			size: int. size of the vector.

		Returns: 
			vector of size(1, size)
		"""

		words = cat.split(' ')
		vectors = np.zeros((len(words), size))
		for i in range(len(words)):
			vectors[i,:] = self.model[words[i]]

		return np.reshape(np.average(vectors, axis=0), (1, size))

