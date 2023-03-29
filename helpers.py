def count_errors_and_confusion_matrix(predict_func, classes_list, val_set):
    confusion_matrix = [[0 for _ in classes_list] for i in classes_list]

    errors = 0
    for sample in val_set:
        prediction = predict_func(sample[:-1])
        if prediction != sample[-1]:
            errors += 1
        confusion_matrix[classes_list.index(prediction)][classes_list.index(sample[-1])] += 1

    return errors, confusion_matrix


def print_results(classes_list, confusion_matrix, val_set, errors):
    print("prediction\\real ", end="")
    for val in classes_list:
        print(f"{val}", end="\t")
    print("")
    for i, val in enumerate(classes_list):
        print(val, end="\t\t")
        for j, _ in enumerate(classes_list):
            print(confusion_matrix[i][j], end="\t")
        print("")

    print(f"\nAccuracy: {(len(val_set) - errors) / len(val_set)}")
    print(f"Test set size: {len(val_set)}")
    print(f"Errors: {errors}")
