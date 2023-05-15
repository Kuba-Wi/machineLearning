import perceptron as p
import helpers as h

import numpy as np
import pandas as pd


class PerceptronsHolder:
    def __init__(self):
        self.p_setosa = None
        self.p_versicolor = None
        self.p_virginica = None
        self.perceptron_list = []

    def create_trained_perceptron(self, train_set, cls_name):
        train_cp = np.copy(train_set)
        np.place(train_cp[:, -1], train_cp[:, -1] == cls_name, 1000)
        np.place(train_cp[:, -1], train_cp[:, -1] != 1000, 0)
        np.place(train_cp[:, -1], train_cp[:, -1] == 1000, 1)
        prc = p.Perceptron()
        prc.train(train_cp)
        return prc

    # def train(self, train_set):
    #     self.p_setosa = self.create_trained_perceptron(train_set, 'Iris-setosa')
    #     self.p_versicolor = self.create_trained_perceptron(train_set, 'Iris-versicolor')
    #     self.p_virginica = self.create_trained_perceptron(train_set, 'Iris-virginica')

    # def predict(self, sample):
    #     pred_dict = {'Iris-setosa': self.p_setosa.prediction_product(sample),
    #                  'Iris-versicolor': self.p_versicolor.prediction_product(sample),
    #                  'Iris-virginica': self.p_virginica.prediction_product(sample)}
    #     max_key = max(pred_dict, key=pred_dict.get)
    #     if pred_dict[max_key] < 0:
    #         return 'Iris-versicolor'
    #     return max_key

    def predict(self, sample):
        pred_dict = {}
        for i in range(0, 10):
            pred_dict[i] = self.perceptron_list[i].prediction_product(sample)
        max_key = max(pred_dict, key=pred_dict.get)
        return max_key

    def train(self, train_set):
        for i in range(0, 10):
            self.perceptron_list.append(self.create_trained_perceptron(train_set, i))


if __name__ == "__main__":
    train_set = np.array(pd.read_csv("optdigits_train.data", header=None))
    val_set = np.array(pd.read_csv("optdigits_val.data", header=None))

    ph = PerceptronsHolder()
    ph.train(train_set)

    classes_list = list({x[-1] for x in train_set})
    errors, confusion_matrix = h.count_errors_and_confusion_matrix(ph.predict, classes_list, val_set)
    h.print_results(classes_list, len(val_set), confusion_matrix, errors)
