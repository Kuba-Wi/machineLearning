import numpy as np
import helpers as h


class Perceptron:
    def __init__(self):
        self.w = None
        self.learning_rate = 0.5

    def train(self, train_set):
        ITERATION_LIMIT = 100
        train_arr = self.__prepare_arrays(train_set)
        for x in train_arr:
            product = sum(np.multiply(x[:-1], self.w))
            cls = -1 if product < 0 else 1
            iters = 0
            while cls != x[-1] and iters < ITERATION_LIMIT:
                iters += 1
                for i in range(len(self.w)):
                    self.w[i] = self.w[i] + self.learning_rate * (x[-1] - cls) * x[i]
                product = sum(np.multiply(x[:-1], self.w))
                cls = -1 if product < 0 else 1

    def predict(self, sample):
        product = sum(np.multiply(sample, self.w[1:])) + self.w[0]
        return 0 if product < 0 else 1

    def __prepare_arrays(self, train_set):
        self.w = np.zeros(train_set.shape[1])
        x = np.concatenate([np.ones((train_set.shape[0], 1)), train_set], axis=1)
        arr = x[:, -1]
        np.place(arr, arr == 0, -1)
        return x

if __name__ == '__main__':
    train_set = np.loadtxt("banknote_train.data", delimiter=',')
    val_set = np.loadtxt("banknote_val.data", delimiter=',')
    p = Perceptron()
    p.train(train_set)

    classes_list = [0, 1]
    errors, confusion_matrix = h.count_errors_and_confusion_matrix(p.predict, classes_list, val_set)
    h.print_results(classes_list, confusion_matrix, val_set, errors)
