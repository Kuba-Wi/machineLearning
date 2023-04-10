import perceptron as p
import helpers as h

import numpy as np
import pandas as pd


class PerceptronsHolder:
    def __init__(self):
        self.p_setosa = None
        self.p_versicolor = None
        self.p_virginica = None

    def create_trained_perceptron(self, train_set, cls_name):
        train_cp = np.copy(train_set)
        np.place(train_cp[:, -1], train_cp[:, -1] == cls_name, 1)
        np.place(train_cp[:, -1], train_cp[:, -1] != 1, 0)
        prc = p.Perceptron()
        prc.train(train_cp)
        return prc

    def train(self, train_set):
        self.p_setosa = self.create_trained_perceptron(train_set, 'Iris-setosa')
        self.p_versicolor = self.create_trained_perceptron(train_set, 'Iris-versicolor')
        self.p_virginica = self.create_trained_perceptron(train_set, 'Iris-virginica')

    def predict(self, sample):
        pred_dict = {'Iris-setosa': self.p_setosa.prediction_product(sample),
                     'Iris-versicolor': self.p_versicolor.prediction_product(sample),
                     'Iris-virginica': self.p_virginica.prediction_product(sample)}
        return max(pred_dict, key=pred_dict.get)


if __name__ == "__main__":
    train_set = np.array(pd.read_csv("iris_train.data", header=None))
    val_set = np.array(pd.read_csv("iris_val.data", header=None))

    ph = PerceptronsHolder()
    ph.train(train_set)

    classes_list = list({x[-1] for x in train_set})
    errors, confusion_matrix = h.count_errors_and_confusion_matrix(ph.predict, classes_list, val_set)
    h.print_results(classes_list, len(val_set), confusion_matrix, errors)
