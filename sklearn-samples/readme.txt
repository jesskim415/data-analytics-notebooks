1. How to run the code .. 

>> pip install -r requirements.txt
>> python predict.py

Run the comment at the bottom to see more options.
>> python predict.py --help


2. How I approached this test ..

The biggest challenge for me was the sparsity of the data.
There were 1000 unique users and 3000 unique books but there were only 2909 user-events that were related to the users and books.
(HOW TO IMPROVE 1: Use user events related to users OR books.)

Also, out of seven features of user and book, five were categorical which made getting the features harder.

So I decided to try four different methods:

-	Simple KNN model with word2vec
-	Classifier and linear regression with word2vec feature vector
-	Neural Network and linear regression with word2vec feature vector
- 	Collective Filtering

3. Models

	A. K Nearest Neighbors with word2vec

	Categorical features of user and categorical features of book are trained separately with word2vec.
	User features and book features are concatenated.
	For each user and book combination, five closest neighborhoods are considered and the score is calculated based on similarity.

	B. Classifier and linear regression with word2vec feature vector

	Categorical features of user and categorical features of book are trained separately with word2vec.
	(Note. I tried one hot vector with user location, but given my cpu capacity, I could not test it.)
	User features and book features are concatenated.

	Classic sklearn classifier is fitted to the impression category.
	Then, the calculated probability to category is again fitted to linear regression.
	(I made it two steps to see if I want to use different values for impression_to_score.)

	C. [CACHED] Neural Network model with word2vec feature vector

	Categorical features of user and categorical features of book are trained separately with word2vec.
	(Note. I tried one hot vector with user location, but given my cpu capacity, I could not test it.)
	User features and book features are concatenated.

