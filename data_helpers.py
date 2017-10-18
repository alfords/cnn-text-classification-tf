import numpy as np
import re
import itertools
from collections import Counter


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_SST_2(train_file, dev_file, test_file):
    '''
    :param train_file:
    :param dev_file:
    :param test_file:
    :return:
    '''
    train_data = list(open(train_file, 'r').readlines())
    dev_data = list(open(dev_file, 'r').readlines())
    test_data = list(open(test_file, 'r').readlines())
    positive_train_data, negative_train_data = _extract_label_and_data(train_data)
    positive_dev_data, negative_dev_data = _extract_label_and_data(dev_data)
    positive_test_data, negative_test_data = _extract_label_and_data(test_data)
    # Data
    x_train = positive_train_data + negative_train_data
    x_train = [clean_str_sst(sent) for sent in x_train]
    x_dev = positive_dev_data + negative_dev_data
    x_dev = [clean_str_sst(sent) for sent in x_dev]
    x_test = positive_test_data + negative_test_data
    x_test = [clean_str_sst(sent) for sent in x_test]
    # Label
    y_train = np.concatenate([[[0, 1] for _ in positive_train_data], [[1, 0] for _ in negative_train_data]], 0)
    y_dev = np.concatenate([[[0, 1] for _ in positive_dev_data], [[1, 0] for _ in negative_dev_data]], 0)
    y_test = np.concatenate([[[0, 1] for _ in positive_test_data], [[1, 0] for _ in negative_test_data]], 0)
    return [x_train, y_train, x_dev, y_dev, x_test, y_test]


def _extract_label_and_data(text):
    positive_examples, negative_examples = [], []
    for sentence in text:
        if sentence[0] == '1':
            positive_examples.append(sentence[1:].strip())
        elif sentence[0] == '0':
            negative_examples.append(sentence[1:].strip())
    return positive_examples, negative_examples


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# if __name__ == '__main__':
#     load_SST_2('./data/SST-2/sst2.train', './data/SST-2/sst2.dev', './data/SST-2/sst2.test')