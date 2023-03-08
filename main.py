from classification import Knn


def get_dataset_from_file(filename):
    dataset = []
    with open(filename) as file:
        for line in file:
            if line == "\n":
                break

            items = line.split(',')
            for i in range(len(items) - 1):
                items[i] = float(items[i])
            dataset.append(items)
    return dataset


train_set = get_dataset_from_file("iris_train.data")
val_set = get_dataset_from_file("iris_val.data")
classes_list = list({x[-1] for x in train_set})
confusion_matrix = [[0 for _ in classes_list] for i in classes_list]

knn = Knn(3)
knn.set_features(train_set)
errors = 0
for element in val_set:
    prediction = knn.predict(element[:-1])
    if prediction != element[-1]:
        errors += 1
    confusion_matrix[classes_list.index(prediction)][classes_list.index(element[-1])] += 1

print("prediction\\real ", end="")
for name in classes_list:
    print(f"{name[:-1]}", end=" ")
print("")
for i, name in enumerate(classes_list):
    print(name[:-1], end="\t\t")
    for j, _ in enumerate(classes_list):
        print(confusion_matrix[i][j], end="\t")
    print("")


print(f"\nAccuracy: {(len(val_set)-errors)/len(val_set)}")
print(f"Test set size: {len(val_set)}")
print(f"Errors: {errors}")
