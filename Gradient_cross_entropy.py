import numpy as np
import pandas as pandas

class Batch_Gradient_descent:

    def sigmoid(self, z):

        s = 1 / (1 + np.exp(-z))

        return s

    def initialize_with_zeros(self, dim):

        w = np.zeros((dim, 1))
        b = 0

        assert (w.shape == (dim, 1))
        assert (isinstance(b, float) or isinstance(b, int))

        return w, b

    def propagate(self, w, b, X, Y):


        m = X.shape[1]

        # compute activation
        A = self.sigmoid(np.dot(w.T, X) + b)

        # compute cost
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = 1 / m * (np.dot(X, (A - Y).T))
        db = 1 / m * (np.sum(A - Y))

        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost=False):


        costs = []

        for i in range(num_iterations):

            # Cost and gradient calculation
            grads, cost = self.propagate(w, b, X, Y)

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule
            w = w - np.dot(learning_rate, dw)
            b = b - np.dot(learning_rate, db)

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

    def predict(self, w, b, X):


        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        # Compute vector "A" predicting the probabilities
        A = self.sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):

            # Convert probabilities A[0,i] to actual predictions p[0,i]
            if A[0, i] <= 0.5:
                Y_prediction[0, i] = 0
            else:
                Y_prediction[0, i] = 1

        assert (Y_prediction.shape == (1, m))

        return Y_prediction

    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

        # initialize parameters with zeros
        w, b = self.initialize_with_zeros(X_train.shape[0])

        # Gradient descent
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate,
                                                 print_cost=print_cost)

        # Retrieve parameters w and b from dictionary "parameters"
        w = parameters["w"]
        b = parameters["b"]

        # Predict test/train set examples (≈ 2 lines of code)
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)

        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return d
