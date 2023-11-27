from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import numpy as np
from tqdm import tqdm


class Evaluator:
    def __init__(self, black_box, surrogate,
                 # x_train, y_train,
                 x_test, y_test):
        self.black_box = black_box  # trained model
        self.surrogate = surrogate  # trained linear surrogate
        # self.x_train = x_train
        # self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def evaluate_bb_predictions(self):
        self.black_box.evaluate(self.x_test, self.y_test, verbose=0)

    def fidelity(self, sklearn_surrogate=False):
        bb_predictions = self.black_box.predict(self.x_test, verbose=0).flatten()
        if sklearn_surrogate:
            sur_predictions = self.surrogate.predict(self.x_test).flatten()
        else:
            sur_predictions = self.surrogate.predict(self.x_test, verbose=0).flatten()

        f = (sur_predictions - bb_predictions) ** 2
        return sum(f) / len(f)

    def accuracy(self, threshold=0.5):
        return accuracy_score(np.array(self.y_test).flatten(),
                              np.where(self.black_box.predict(self.x_test, verbose=0).flatten() >= threshold, 1, 0))

    def f1(self, threshold=0.5, average='macro'):
        return f1_score(np.array(self.y_test).flatten(),
                        np.where(self.black_box.predict(self.x_test, verbose=0).flatten() >= threshold, 1, 0),
                        average=average)

    def r2_agreement(self, sklearn_surrogate=False):
        if sklearn_surrogate:
            sur_predictions = self.surrogate.predict(self.x_test).flatten()
        else:
            sur_predictions = self.surrogate.predict(self.x_test, verbose=0).flatten()
        return r2_score(self.black_box.predict(self.x_test, verbose=0).flatten(),
                        sur_predictions)

    def mse(self):
        return mean_squared_error(np.array(self.y_test).flatten(),
                                  self.black_box.predict(self.x_test, verbose=0).flatten())

    def r2(self):
        return r2_score(np.array(self.y_test).flatten(), self.black_box.predict(self.x_test, verbose=0).flatten())

    def local_fidelity(self, train_points, test_points, e, surrogate=None,
                       fit_surrogate=True, num_neighbors=10, use_tqdm=True,
                       predict_function=None, categorical_features=None):
        if predict_function is None:
            predict_function = lambda z: self.black_box.predict(z, verbose=0).flatten()
        total_local_fidelity = 0.
        iterator = test_points
        if use_tqdm:
            iterator = tqdm(test_points)
        for x in iterator:
            explanation = e.explain_instance(x, predict_function, save_surrogate=False,
                                             model_regressor=surrogate, fit_surrogate=fit_surrogate,
                                             save_neighbors=False)
            neighbors = list()
            for _ in range(num_neighbors):
                noise = 0.1 * np.random.normal(loc=0.0, scale=1.0, size=x.shape)  # continuous noise
                neighbor = x + noise
                if categorical_features is not None:
                    for feature in categorical_features:
                        neighbor[feature] = np.random.choice(np.unique(test_points[:, feature]))  # categorical noise
                neighbors.append(neighbor)

            neighbors = np.array(neighbors)

            weights = np.zeros(shape=x.shape)
            coefficients_pairs = explanation.local_exp[1]
            for pair in coefficients_pairs:
                weights[pair[0]] = pair[1]
            weights = weights / np.sqrt(np.var(train_points, axis=0))
            intercept = explanation.intercept[1] - np.sum(weights * np.mean(train_points, axis=0))
            weights = np.insert(weights, 0, intercept)  # add bias
            g = np.dot(np.insert(neighbors, 0, 1, axis=1), weights).flatten()

            f = self.black_box.predict(neighbors, verbose=0).flatten()  # len(neighbors)
            fid = (g - f) ** 2
            total_local_fidelity += np.mean(fid)  # neighborhood fidelity

        return total_local_fidelity / len(test_points)  # average of neighborhoods
