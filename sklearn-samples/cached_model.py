from utils import Dataset
import numpy as np 

class CollaborativeFiltering:
	def __init__(self, rating, user_mu, num_features):
		self.rating = rating
		self.user_mu = user_mu

		self.nonzero_indices = np.transpose(np.nonzero(self.rating))
		self.rating -= self.user_mu

		self.rating = np.nan_to_num(self.rating)

		self.num_features = num_features
		
		self.user_param = np.random.rand(self.rating.shape[0], num_features) * 1e-4
		self.book_param = np.random.rand(self.rating.shape[1], num_features) * 1e-4

		self.loss_v = np.zeros((self.nonzero_indices.shape[0], 1))

		print(self.rating)

	def _loss(self, user_index, book_index):
		return np.asscalar(np.dot(self.user_param[user_index,:], self.book_param[book_index,:].T)) - self.rating[user_index, book_index]

	def _user_reg(self, user_index, reg):
		return np.sum(self.user_param[user_index] ** 2) * reg 

	def _book_reg(self, book_index, reg):
		return np.sum(self.book_param[book_index] ** 2) * reg 

	def user_gradient(self, user_index, book_index, reg_coef):
		"""
		Parameter:
			user - (1 * num_features) feature vector representing user
			book - (1 * num_features) feature vector representing book
			rating - int. rating of user and book.
			reg - regularization term for the feature
		"""
		grad = np.asscalar(np.dot(self.user_param[user_index,:], self.book_param[book_index,:].T)) - rating[user_index, book_index]
		row_grad = grad * self.book_param[book_index]
		regularization = self.user_param[user_index]

		user_grad = row_grad + reg_coef * regularization

		return user_grad

	def book_gradient(self, user_index, book_index, reg_coef):
		"""
		Parameter:
			user - (1 * num_features) feature vector representing user
			book - (1 * num_features) feature vector representing book
			rating - int. rating of user and book.
			reg - regularization term for the feature
		"""
		grad = np.asscalar(np.dot(self.user_param[user_index], self.book_param[book_index].T)) - rating[user_index, book_index]
		row_grad = grad * self.user_param[user_index]
		regularization = self.book_param[book_index]

		book_grad = row_grad + reg_coef * regularization

		return book_grad 	

	def train(self, lr, iteration, user_reg, book_reg):
		for i in range(iteration):
			row_loss = {}
			col_loss = {}
			for j in range(self.nonzero_indices.shape[0]):
				self.loss_v[j,:] = self._loss(self.nonzero_indices[j,0], self.nonzero_indices[j,1]) 
				if self.nonzero_indices[j,0] not in row_loss.keys():
					row_loss[self.nonzero_indices[j,0]] = self.loss_v[j]
				else:
					row_loss[self.nonzero_indices[j,0]] += self.loss_v[j]
			if self.nonzero_indices[j,1] not in col_loss.keys():
					col_loss[self.nonzero_indices[j,1]] = self.loss_v[j]
				else:
					col_loss[self.nonzero_indices[j,1]] += self.loss_v[j]

			loss = np.sum(self.loss_v ** 2)
			user_regularization = user_reg * np.sum(self.user_param ** 2)
			book_regularization = book_reg * np.sum(self.book_param ** 2)

			total_loss = loss + user_regularization + book_regularization

			print('Total loss at iteration %d: %f' % (i, total_loss))



class DLClassifier:
    def __init__(self, hidden_layers, num_input, num_classes):

        self.weights = {
            'w1':  tf.Variable(tf.random_normal([num_input, hidden_layers[0]]), name='W1'),
            'w2':  tf.Variable(tf.random_normal([hidden_layers[0], hidden_layers[1]]), name='W2'),
            'w3':  tf.Variable(tf.random_normal([hidden_layers[1], hidden_layers[2]]), name='W3'),
            'w4':  tf.Variable(tf.random_normal([hidden_layers[2], num_classes]), name='W4')
        }

        self.biases = {
            'b1':  tf.Variable(tf.random_normal([hidden_layers[0]]), name='b1'),
            'b2':  tf.Variable(tf.random_normal([hidden_layers[1]]), name='b2'),
            'b3':  tf.Variable(tf.random_normal([hidden_layers[2]]), name='b3'),
            'b4':  tf.Variable(tf.random_normal([num_classes]), name='b4')
        }

        self.num_input = num_input
        self.num_classes = num_classes

        #self.x = tf.placeholder(tf.float32, [None, num_input], name='InputData')
        #self.y = tf.placeholder(tf.float32, [None, num_classes], name='InputData')
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self._index = 0

    def get_data(self, filepath, val_rate=0.01):
        X, Y = load_dataset(filepath)
        X += 1e-8
        #Y_one_hot = tf.one_hot(Y, depth=self.num_classes)
        #print(Y_one_hot)
        ids = np.reshape(range(1, self.num_classes + 1), (self.num_classes, 1))
        enc = OneHotEncoder(handle_unknown='ignore')    
        enc.fit(ids)
        Y_oh = enc.transform(np.reshape(Y, (len(Y), 1))).toarray()
        self.X_train, self.X_test, self.Y_train, self.Y_test = split_dataset(X, Y_oh, val_rate=val_rate)


    def get_next_batch(self, batch_size, idx):
        """
        Return a total of `num` random samples and labels. 
        """
        start = self._index
        self._index = min(self._index + batch_size, self.X_train.shape[0])
        end = self._index

        batch_idx = idx[start:end]
        data_shuffle = self.X_train[batch_idx,:]
        labels_shuffle = self.Y_train[batch_idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

 
# Create model
    def neural_net(self, x, weights, biases):

        layer_1 = tf.add(tf.matmul(x, self.weights['w1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        #tf.summary.histogram("relu1", layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, self.weights['w2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        #tf.summary.histogram("relu2", layer_2)

        layer_3 = tf.add(tf.matmul(layer_2, self.weights['w3']), self.biases['b3'])
        layer_3 = tf.nn.relu(layer_3)
        #tf.summary.histogram("relu3", layer_3)
        # Output layer

        out_layer = tf.add(tf.matmul(layer_3, self.weights['w4']), self.biases['b4'])
        return out_layer


    def train(self, learning_rate=0.0001, training_epochs=1000, batch_size=128, display_step=1, logs_path='./logs'):
        x = tf.placeholder(tf.float32, [None, self.num_input], name='InputData')
        y = tf.placeholder(tf.float32, [None, self.num_classes], name='InputData')

        print(self.X_train)
        with tf.name_scope('Model'):
            pred = self.neural_net(x, self.weights, self.biases)

        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

        with tf.name_scope('SGD'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

        with tf.name_scope('Accuracy'):
            acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))

        init = tf.global_variables_initializer()
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", acc)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        #for grad, var in grads:
        #    tf.summary.histogram(var.name + '/gradient', grad)
        merged_summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(init)

            summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())

            for epoch in range(training_epochs):
                avg_cost = 0.
                #total_batch = int(mnist.train.num_examples/batch_size)
                total_batch = int(self.X_train.shape[0] / batch_size)

                _idx = np.arange(0, self.X_train.shape[0])
                np.random.shuffle(_idx)
                self._index = 0
                for i in range(total_batch):
                    batch_xs, batch_ys = self.get_next_batch(batch_size, _idx)
                    _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
                    summary_writer.add_summary(summary, epoch * total_batch + i)
                    avg_cost += c / total_batch

                if (epoch+1) % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

            print("Optimization Finished!")

            print("Accuracy:", acc.eval({x: self.X_test, y: self.Y_test}))







