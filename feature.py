# Hayden Moore
# Intro to Machine Learning 2020
# feature.py

import sys

def get_data_set(path, dict, feature_flag):
    threshold = 4
    dict_words = {}
    feature_representation = []

    # get dict words
    with open(dict, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            dict_words.update({line[0]:line[1]})

    # read in the data
    with open(path, 'r') as f:
        lines = f.readlines()

    # build the feature representation
    i = 0
    # feature flag 1
    if feature_flag == '1':
        for line in lines:
            lines[i] = line.split()
            feature_representation.append({"label": int(lines[i][0]), "features":[]})
            dict_set = set(dict_words)
            for word in lines[i]:
                if word in dict_set:
                    index = dict_words[word]
                    feature_representation[i]["features"].append(str(index) + ':1')
                    dict_set.remove(word)
            i += 1

    # feature flag 2
    if feature_flag == '2':
        for line in lines:
            used_words_count = {}
            lines[i] = line.split()
            feature_representation.append({"label": int(lines[i][0]), "features":[]})
            dict_set = set(dict_words)
            for word in lines[i]:
                if word in dict_set:
                    index = dict_words[word]
                    try:
                        used_words_count[str(index)] += 1
                        if used_words_count[str(index)] >= threshold:
                            f_index = feature_representation[i]["features"].index(str(index) + ':1')
                            feature_representation[i]["features"].pop(f_index)
                            dict_set.remove(word)
                    except:
                        feature_representation[i]["features"].append(str(index) + ':1')
                        used_words_count.update({str(index): 1})
            i += 1
    return feature_representation


def write_tsv(output, actual_data):
    # output feature representation
    with open(output, 'w') as f:
        for data in actual_data:
            f.write("%s" % data["label"])
            for feature in data["features"]:
                f.write("\t%s" % feature)
            f.write('\n')


def main():
    train_data = sys.argv[1]
    valid_data = sys.argv[2]
    test_data = sys.argv[3]
    dict = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_valid_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = sys.argv[8]

    # build feature representation
    train_data = get_data_set(train_data, dict, feature_flag)
    valid_data = get_data_set(valid_data, dict, feature_flag)
    test_data = get_data_set(test_data, dict, feature_flag)

    # write output tsv files
    write_tsv(formatted_train_out, train_data)
    write_tsv(formatted_valid_out, valid_data)
    write_tsv(formatted_test_out, test_data)

if __name__ == "__main__":
    main()

