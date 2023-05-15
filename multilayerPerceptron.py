from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import helpers as h

if __name__ == '__main__':
    train_set = np.array(pd.read_csv("optdigits_train.data", header=None))
    val_set = np.array(pd.read_csv("optdigits_val.data", header=None))

    model = MLPClassifier()
    model = model.fit(train_set[:, :-1], train_set[:, -1])

    classes_list = list({x[-1] for x in train_set})
    errors, confusion_matrix = h.count_errors_and_confusion_matrix_sklearn(model.predict, classes_list, val_set)
    h.print_results(classes_list, len(val_set), confusion_matrix, errors)
