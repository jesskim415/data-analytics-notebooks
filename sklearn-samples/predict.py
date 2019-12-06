from knearneighborhood import KNNModel
from classifier import Classifier
from utils import Dataset
import sys
import argparse


def main(args):
	dataset = Dataset()
	dataset.get_ratings_from_user_events()
	dataset.get_user_features(c2v_file=args.user_c2v)
	dataset.get_book_features(features=args.book_features.split(','), c2v_file=args.book_c2v)
	dataset.get_ue_features()

	if args.mode == 'knn':
		knn = KNNModel(dataset.user_features, dataset.book_features, dataset.ue_features, dataset.ratings, dataset.known_index)
		prediction = knn.predict_rating(k=args.k)
		knn.save_to_csv(out=args.out, userIds=dataset.user_df['user'].values, bookIds=dataset.book_df['bookISBN'].values)
	elif args.mode == 'classifier':
		classifier = Classifier(dataset.user_features, dataset.book_features, dataset.ue_features, dataset.ratings, dataset.known_index)
		classifier.get_dataset()
		classifier.train(clf=args.clf)
		classifier.to_logistic_regression()
		classifier.predict_rating()
		classifier.save_to_csv(out=args.out, userIds=dataset.user_df['user'].values, bookIds=dataset.book_df['bookISBN'].values)
	elif args.mode == 'more':
		print('New models (collaborative filtering and DNN) are under development. Please take a look at "cached_model.py" ')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='knn', choices=['knn','classifier','more'], help='Prediction method.')
    parser.add_argument('--user_c2v', default='./trained_model/user_c2v', help='Path for user cat2vec file.')
    parser.add_argument('--book_c2v', default='./trained_model/book_c2v', help='Path for book cat2vec file.')
    parser.add_argument('--book_features', default='bookName,author,publisher,yearOfPublication,urlId', 
    			help='List of the features to be used for book feature calculation. Please make sure they are separated by commas with no space')
    parser.add_argument('--k', default=5, type=int, help='Number of near neighborhoods to be used during prediction')
    parser.add_argument('--clf', default='random_forest', choices=['decision_tree','random_forest', 'mlp', 'adaboost'], 
    			help='The kind of classifier to be used in Classifer()')
    parser.add_argument('--out', default='out.csv', help='Path of output file')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


