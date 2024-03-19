# coding=utf-8
import numpy
import csv
import math
import collections
import itertools
import time
import csv

import numpy
import pandas as pd
from sklearn import model_selection as skms
import matplotlib.pyplot as plt

from c45 import Node
from c45 import DecisionTree
from c45 import MetricGenerator

DATA_SET = "subsets_variable_length/subset_1_34_df.csv"
TRAINING_SET = "data_set/subset_1_34_tr.csv"
TESTING_SET = "data_set/subset_1_34_te.csv"

def create_valid_identifier(header):
        # Convert spaces to underscores and remove other invalid characters
        return ''.join(c if c.isalnum() or c == '_' else '_' for c in header)

def iter_once(tr_size=0.5, seed=1, min_gain=0, min_num=2, max_depth=100, verbose=False):
    # split dataset into training set and testing set.
    data_set = pd.read_csv(DATA_SET)
    training_set, testing_set = skms.train_test_split(
        data_set, train_size=tr_size, random_state=seed)
    training_set.to_csv(TRAINING_SET, index=False)
    testing_set.to_csv(TESTING_SET, index=False)
    # if verbose:
    #     print("training set: \n", training_set.describe())
    #     print("testing set: \n", testing_set.describe())
    #     print("data set: \n", data_set.describe())

    # initialize decision tree, and pull entries in the TRAINING_SET into root node.
    tree = DecisionTree(TRAINING_SET)

    with open(DATA_SET, "r", encoding="utf-8") as file:
            f_csv = csv.reader(file)
            headers = next(f_csv)
            identifiers = [create_valid_identifier(header) for header in headers]
            i=0
            for row in f_csv:
                    if i < len(identifiers):
                        m = tree.generator.contineous_template(identifiers[i], *row)
                    else:
                        # Handle case where identifiers are exhausted
                        break
                    i += 1
    # we need all values's ordered set of each feature, to generate metric functions.

    # min gain ratio should every splitting get,
    # min number of items supporting splitting,
    # max tree depth, considering root node.
    tree.set_super_parameters(min_gain, min_num, max_depth)
    tree.train()
    error_rate_train = 1 - tree.run_test(TRAINING_SET)
    error_rate_test = 1 - tree.run_test(TESTING_SET)
    if verbose:
        print("error rate on trainning set: {:.4f}% || testing set: {:.4f}%".format(
            100 * error_rate_train, 100 * error_rate_test))
    return error_rate_train, error_rate_test


def cross_validation():
    # cross validation
    tr_res = []
    te_res = []
    for i in range(500):
        now = int(time.time()*100) % 1000000  # random seed
        err_tr, err_te = iter_once(
            tr_size=0.5, seed=now, min_gain=0.2, min_num=4, max_depth=6, verbose=True)
        tr_res.append(err_tr)
        te_res.append(err_te)
    tr_mean = sum(tr_res) / len(tr_res)
    te_mean = sum(te_res) / len(te_res)
    tr_max = max(tr_res)
    te_max = max(te_res)
    tr_min = min(tr_res)
    te_min = min(te_res)
    print("\nresult of cross validation:")
    print("{:<16s}{:<10s}{:<10s}{:<10s}".format("", "min", "max", "average"))
    print("{:<16s}{:<10f}{:<10f}{:<10f}".format(
        "training set", tr_min, tr_max, tr_mean))
    print("{:<16s}{:<10f}{:<10f}{:<10f}".format(
        "testing set", te_min, te_max, te_mean))


def para_effect_min_gain():
    tr_res = []
    te_res = []
    for i in range(100):
        min_gain = i*0.05
        err_tr, err_te = iter_once(
            tr_size=0.7, seed=1, min_gain=min_gain, verbose=True)
        tr_res.append(err_tr)
        te_res.append(err_te)

    x = numpy.linspace(0, 20, 100)
    plt.figure(num=1, figsize=(8, 5))
    plt.plot(x, tr_res, color='blue', linewidth=1.0, label="trainning set")
    plt.plot(x, te_res, color="green", linewidth=1.0,
             linestyle="-.", label="testing set")
    plt.xlim((0, 20))
    plt.xlabel("min gain ratio")
    plt.ylabel("error rate")
    plt.legend()


def para_effect_min_num():
    tr_res = []
    te_res = []
    for i in range(2, 50):
        min_num = i
        err_tr, err_te = iter_once(
            tr_size=0.7, seed=1, min_num=min_num, verbose=True)
        tr_res.append(err_tr)
        te_res.append(err_te)

    x = numpy.linspace(2, 50, 48)
    plt.figure(num=2, figsize=(8, 5))
    plt.plot(x, tr_res, color='blue', linewidth=1.0, label="trainning set")
    plt.plot(x, te_res, color="green", linewidth=1.0,
             linestyle="-.", label="testing set")
    plt.xlim((2, 50))
    plt.xlabel("min items number")
    plt.ylabel("error rate")
    plt.legend()


def para_effect_max_depth():
    tr_res = []
    te_res = []
    for i in range(2, 20):
        max_depth = i
        err_tr, err_te = iter_once(
            tr_size=0.7, seed=1, max_depth=max_depth, verbose=True)
        tr_res.append(err_tr)
        te_res.append(err_te)

    x = numpy.linspace(2, 20, 18)
    plt.figure(num=3, figsize=(8, 5))
    plt.plot(x, tr_res, color='blue', linewidth=1.0, label="trainning set")
    plt.plot(x, te_res, color="green", linewidth=1.0,
             linestyle="-.", label="testing set")
    plt.xlim((2, 20))
    plt.xlabel("max depth")
    plt.ylabel("error rate")
    plt.legend()


if __name__ == "__main__":
    cross_validation()
    para_effect_max_depth()
    plt.show()
