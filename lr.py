# Hayden Moore
# Intro to Machine Learning 2020
# lr.py

import sys
import math

def dot_product(theta, x):
    product = 0.0
    for i in x:
        product += theta[i] * 1
    return product

def sigmoid(a):
    sig = 1 / (1 + math.exp(-a))
    return sig

def predict(a):
    if a >= 0.5:
        return 1
    else:
        return 0


def train(formatted_input, dict, num_epoch):
    # get dict
    with open(dict, 'r') as f:
        lines = f.readlines()

    # split dict
    dict = []
    for line in lines:
        line = line.split()
        dict.append(line[0])

    # get formatted input
    with open(formatted_input, 'r') as f:
        lines = f.readlines()

    # split/sort data set
    i = 0
    all_vectors = []
    for line in lines:
        lines[i] = line.split()
        vector = {"label": int(lines[i].pop(0)), "features": []}
        for word in lines[i]:
            word = word.split(':')
            vector["features"].append(int(word[0]))
        # bias term
        vector["features"].append(39176)
        all_vectors.append(vector)
        i += 1

    # stochastic gradient descent
    # bias term
    theta_len = len(dict) + 1
    theta = [0] * theta_len

    e = 0
    while e < int(num_epoch):
        for vector in all_vectors:
            # actual label
            y = vector["label"]
            # dot product
            dot = dot_product(theta, vector["features"])
            gradient = y - sigmoid(dot)

            # calculate each parameter of theta and update
            for x in vector["features"]:
                # update actual theta
                theta[x] = theta[x] + (0.1 * gradient)
        e += 1
    print(theta)
    return theta


def test(formatted_input, theta, output):
    # get formatted input
    with open(formatted_input, 'r') as f:
        lines = f.readlines()

    # split/sort data set
    i = 0
    all_vectors = []
    actual_labels = []
    for line in lines:
        lines[i] = line.split()
        label = int(lines[i].pop(0))
        actual_labels.append(label)
        vector = {"label": label, "features": []}
        for word in lines[i]:
            word = word.split(':')
            vector["features"].append(int(word[0]))
        # bias term
        vector["features"].append(39176)
        all_vectors.append(vector)
        i += 1

    predicted_labels = []
    i = 1
    for vector in all_vectors:
        dot = dot_product(theta, vector["features"])
        label = predict(sigmoid(dot))
        predicted_labels.append(label)
        i += 1

    i = 0
    j = 0
    for a in actual_labels:
        if a != predicted_labels[i]:
            j += 1
        i += 1

    with open(output, 'w') as f:
        for p in predicted_labels:
            f.write(str(p) + '\n')

    return j/i


def main():
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = sys.argv[8]

    # call training function
    theta = train(formatted_train_input, dict_input, num_epoch)

    # call testing function
    test_error = test(formatted_test_input, theta, test_out)
    train_error = test(formatted_train_input, theta, train_out)

    # write metrics output
    with open(metrics_out, 'w') as f:
        f.write("error(train): " + str(train_error) + '\n')
        f.write("error(test): " + str(test_error))


if __name__ == "__main__":
    main()