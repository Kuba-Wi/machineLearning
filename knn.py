import helpers as h


def get_dataset_from_file(filename):
    dataset = []
    with open(filename) as file:
        for line in file:
            if line == "\n":
                break

            items = line.split(',')
            for i in range(len(items) - 1):
                items[i] = float(items[i])
            items[-1] = items[-1].removesuffix("\n")
            dataset.append(items)
    return dataset


class Knn:
    def __init__(self, n_neighbours):
        self.n_neighbours = n_neighbours
        self.features = []

    def set_features(self, features):
        self.features = features

    def predict(self, x_test):
        distances = []
        for index, feature in enumerate(self.features):
            dist = 0
            for i, attr in enumerate(x_test):
                dist += (attr - feature[i])**2
            distances.append((feature[-1], dist))   # (class, distance)

        class_count = {}    # class: number of neighbours
        for i in range(self.n_neighbours):
            cls, dist = min(distances, key=lambda pair: pair[1])
            class_count[cls] = class_count.get(cls, 0) + 1
            distances.pop(distances.index((cls, dist)))

        return max(class_count, key=class_count.get)


if __name__ == "__main__":
    train_set = get_dataset_from_file("iris_train.data")
    val_set = get_dataset_from_file("iris_val.data")
    knn = Knn(3)
    knn.set_features(train_set)

    classes_list = list({x[-1] for x in train_set})
    errors, confusion_matrix = h.count_errors_and_confusion_matrix(knn.predict, classes_list, val_set)

    h.print_results(classes_list, len(val_set), confusion_matrix, errors)
