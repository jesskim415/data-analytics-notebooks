import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import sys


class KNNModel:
	def __init__(self, user_features, book_features, ue_features, ratings, known_index, metric='cosine', algorithm='brute'):
		self.user_features = user_features
		self.book_features = book_features
		self.ue_features = ue_features
		self.ratings = ratings
		self.known_index = known_index
		self.ue_features = ue_features

		self.nonzero_indices = np.transpose(np.nonzero(self.ratings)).tolist()

		self.user_knn = NearestNeighbors(metric=metric, algorithm=algorithm)
		self.user_knn.fit(self.user_features)

		self.book_knn = NearestNeighbors(metric=metric, algorithm=algorithm)
		self.book_knn.fit(self.book_features)

		self.prediction = None


	def get_k_similar(self, idx, kind, k, metric='cosine'):
		'''
		'''
		if kind == 'user':
			features = self.user_features
			knn = self.user_knn
		elif kind == 'book':
			features = self.book_features
			knn = self.book_knn
		#elif kind == 'ue':
		#    features = np.r_[feature, self.ue_features]

		distances, indices = knn.kneighbors(np.reshape(features[idx,:], (1, features.shape[1])), n_neighbors = k+1)
		similarities = 1-distances.flatten()

		return similarities[1:,], indices[:,1:]

	def get_item_based_prediction(self, user_idx, book_idx, k=100, metric='cosine'):
		"""
		Parameter:
			user_idx - int. User row number.
			book_idx - int. Book row number.
			k - int. Number of neighborhoods
		Returns:
			simrate - vector of size (k,)
		Notes:
			This is not used in actual production of output. This is to give an insight to the process.
		"""

		user_similarity_score, sim_users = self.get_k_similar(idx=user_idx, kind='user', k=k)
		book_similarity_score, sim_books = self.get_k_similar(idx=book_idx, kind='book', k=k)

		sim_users = np.reshape(sim_users, (k,)).tolist()
		sim_books = np.reshape(sim_books, (k,)).tolist()

		known_index = [tuple(index) for index in self.known_index]

		similar_ues = [(i, j) for i in range(k) for j in range(k) 
			if (sim_users[i], sim_books[j]) in known_index]

		ratings = [self.ratings[similar_ues[i][0],similar_ues[i][1]] for i in range(k)]
		ratings = sum(ratings) / k		

		return ratings

	def predict_by_item(self, idx, kind, k, metric='cosine'):
		"""
		Parameter:
			user_idx: int. User row number.
			k: int. Number of nearest neighborhood to be calculated.
		Return:
			prediction: array of size 1 * num_books.

		"""
		if kind == 'user':
			features = np.c_[np.tile(self.user_features[idx,:], (self.book_features.shape[0], 1)), self.book_features]
		elif kind == 'book':
			features = np.c_[self.user_features, np.tile(self.book_features[idx,:], (self.user_features.shape[0], 1))]

		print("Calculating prediction for %s #%d" % (kind, idx))
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")

		# ranks: similarity of features to the known ue_features.
		# ind: for each row, it sorts out the k maximum ranks and returns the indices
		# 	ind gives row number of known index.
		# 	known_index[ind[i,], 0] is the user corresponding to ue_features of ind[i]
		# 	known_index[ind[i,], 1] is the book corresponding to ue_features of ind[i]

		ranks = cosine_similarity(features, self.ue_features)
		ind = np.argpartition(ranks, -k)[:,-k:]

		top_k_similar_ranking = np.asarray([ranks[i, ind[i]] for i in range(ranks.shape[0])])
		known_index = np.asarray(self.known_index)
		top_k_scores = np.asarray([self.ratings[known_index[ind[i,], 0], known_index[ind[i,], 1]] for i in range(ranks.shape[0])])

		prediction = np.einsum('ij,ij->i', top_k_similar_ranking, top_k_scores) / np.sum(top_k_similar_ranking, axis=1)

		if kind == 'user':
			prediction = np.reshape(prediction, (1, prediction.shape[0]))

		return prediction


	def predict_rating(self, k, kind='user', metric='cosine'):
		"""
		Loop over users or books to obtain all predictions
		"""
		self.prediction = np.zeros((self.user_features.shape[0], self.book_features.shape[0]))

		if kind == 'user':
			for i in range(self.prediction.shape[0]):
				self.prediction[i,:] = self.predict_by_item(idx=i, kind='user', k=k)

		elif kind == 'book':
			for j in range(self.prediction.shape[1]):
				self.prediction[:,j] = self.predict_by_item(idx=j, kind='book', k=k)

		return self.prediction

	def save_to_csv(self, out, userIds, bookIds):
		for index in self.known_index:
			self.prediction[index[0], index[1]] = self.ratings[index[0], index[1]]

		self.prediction = (self.prediction * 10).astype(int)

		df = pd.DataFrame(self.prediction, index=userIds, columns=bookIds)
		df.to_csv(out)

		return None