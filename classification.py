from random import randrange


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
