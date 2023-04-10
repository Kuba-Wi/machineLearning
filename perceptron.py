import numpy as np
import helpers as h


class Perceptron:
    def __init__(self):
        self.w = None
        self.learning_rate = 0.1

    def train(self, train_set):
        ITERATION_LIMIT = 100
        self.w = np.zeros(train_set.shape[1])
        train_array = np.concatenate([np.ones((train_set.shape[0], 1)), train_set], axis=1)
        for _ in range(ITERATION_LIMIT):
            for x in train_array:
                product = np.dot(x[:-1], self.w)
                cls = 0 if product < 0 else 1
                if cls != x[-1]:
                    for i in range(len(self.w)):
                        self.w[i] += self.learning_rate * (x[-1] - cls) * x[i]

    def prediction_product(self, sample):
        return np.dot(sample, self.w[1:]) + self.w[0]

    def predict(self, sample):
        return 0 if self.prediction_product(sample) < 0 else 1


if __name__ == '__main__':
    train_set = np.loadtxt("banknote_train.data", delimiter=',')
    val_set = np.loadtxt("banknote_val.data", delimiter=',')
    p = Perceptron()
    p.train(train_set)

    classes_list = [0, 1]
    errors, confusion_matrix = h.count_errors_and_confusion_matrix(p.predict, classes_list, val_set)
    h.print_results(classes_list, len(val_set), confusion_matrix, errors)
