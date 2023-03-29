from random import randrange


def split_train_val_test(data):
    initial_length = len(data)
    validation_set = []
    for _ in range(int(initial_length * 0.2)):
        validation_set.append(data.pop(randrange(len(data))))
    test_set = []
    for _ in range(int(initial_length * 0.2)):
        test_set.append(data.pop(randrange(len(data))))

    return data, validation_set, test_set


def write_to_file(filename, data):
    with open(filename, "w") as f:
        for line in data:
            f.writelines(line)


if __name__ == "__main__":
    d = []
    with open("data_banknote_authentication.txt") as file:
        for line in file:
            if line != "\n":
                d.append(line)

    train_set, val_set, test_set = split_train_val_test(d)
    write_to_file("banknote_train.data", train_set)
    write_to_file("banknote_val.data", val_set)
    write_to_file("banknote_test.data", test_set)