import os
import re
import nltk
import math
import numpy as np
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import SnowballStemmer
snow = SnowballStemmer('english')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
train_path = "/Users/nam/Desktop/dataset/dev/"
test_path = "/Users/nam/Desktop/dataset/test/"

class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def load_corpus(self, path):
        corpus = []
        for i in os.listdir(path):
            final = os.walk(path + i).__next__()[0]
            name = final
            size = len(os.walk(final).__next__()[2])
            for j in range(size):
                idx = str(j + 1)
                final_path = name + "/" + i + "_" + idx + ".txt"

                f = open(final_path, 'r')
                data = f.read()

                sentences = []
                for sen in sent_tokenize(data):
                    sentences.append(sen.replace("\n", " "))

                for s in sentences:
                    final = []
                    result = []
                    tokens = TreebankWordTokenizer().tokenize(s)
                    for w in tokens:
                        if w not in stop_words:
                            if w[len(w) - 1] == ".":
                                result.append(w[:len(w) - 1])
                            else:
                                result.append(w)
                    for w in result:
                        word = snow.stem(w)
                        if len(word) > 0:
                            if word[0] == "'" or word[0] == "." or word[0] == "`" or word[0] == "," or word[0] == "(" or \
                                    word[0] == ")" or word[
                                len(word) - 1] == ".":
                                continue
                            else:
                                final.append(word.lower())

                    full = []
                    for add in pos_tag(final):
                        category = (add[0], add[1], i)
                        full.append(category)

                    corpus.append(full)
                f.close()
        return corpus

    def count_words(self, training_set):
        counts = defaultdict(lambda : [0, 0, 0, 0])
        for object in training_set:
            for w in object:
                if w[2] == "interest":
                    if w[0] == "interest":
                        counts[(w[0], w[1])][0] += 10
                    else:
                        counts[(w[0], w[1])][0] += 1
                elif w[2] == "jobs":
                    if w[0] == "job":
                        counts[(w[0], w[1])][1] += 10
                    else:
                        counts[(w[0], w[1])][1] += 1
                elif w[2] == "money_supply":
                    if w[0] == "money_supli":
                        counts[(w[0], w[1])][2] += 10
                    else:
                        counts[(w[0], w[1])][2] += 1
                elif w[2] == "trade":
                    if w[0] == "trade":
                        counts[(w[0], w[1])][3] += 10
                    else:
                        counts[(w[0], w[1])][3] += 1

        return counts

    def word_probabilities(self, counts, total_class0, total_class1, total_class2, total_class3, k):
        return [(w[0],
                 (w[1][0] + k) / (total_class0 + 2 * k),
                 (w[1][1] + k) / (total_class1 + 2 * k),
                 (w[1][2] + k) / (total_class2 + 2 * k),
                 (w[1][3] + k) / (total_class3 + 2 * k))
                 for w in counts.items()]

    def classify_probability(self, word_probs, doc):
        result = []
        final = []
        tokens = TreebankWordTokenizer().tokenize(doc)
        for w in tokens:
            if w not in stop_words:
                if w[len(w) - 1] == ".":
                    result.append(w[:len(w) - 1])
                else:
                    result.append(w)
        for w in result:
            word = snow.stem(w)
            if len(word) > 0:
                if word[0] == "'" or word[0] == "." or word[0] == "`" or word[0] == "," or word[0] == "(" or \
                        word[0] == ")" or word[
                    len(word) - 1] == ".":
                    continue
                else:
                    final.append(word.lower())

        data = pos_tag(final)

        log_prob_if_class0 = log_prob_if_class1 = log_prob_if_class2 = log_prob_if_class3 = 0.0

        for word, prob_if_class0, prob_if_class1, prob_if_class2, prob_if_class3 in word_probs:
            if word in data:
                log_prob_if_class0 += math.log(prob_if_class0)
                log_prob_if_class1 += math.log(prob_if_class1)
                log_prob_if_class2 += math.log(prob_if_class2)
                log_prob_if_class3 += math.log(prob_if_class3)
            else:
                if(prob_if_class0 < 1.0):
                    log_prob_if_class0 += math.log(1.0 - prob_if_class0)
                if(prob_if_class1 < 1.0):
                    log_prob_if_class1 += math.log(1.0 - prob_if_class1)
                if(prob_if_class2 < 1.0):
                    log_prob_if_class2 += math.log(1.0 - prob_if_class2)
                if(prob_if_class3 < 1.0):
                    log_prob_if_class3 += math.log(1.0 - prob_if_class3)

        prob_if_class0 = math.exp(log_prob_if_class0)
        prob_if_class1 = math.exp(log_prob_if_class1)
        prob_if_class2 = math.exp(log_prob_if_class2)
        prob_if_class3 = math.exp(log_prob_if_class3)
        total = prob_if_class0 + prob_if_class1 + prob_if_class2 + prob_if_class3

        return (prob_if_class0/total, prob_if_class1/total, prob_if_class2/total, prob_if_class3/total)



    def train(self, trainfile_path):
        training_set = self.load_corpus(trainfile_path)

        num_class0 = 0
        num_class1 = 0
        num_class2 = 0
        num_class3 = 0
        for i in os.listdir(trainfile_path):
            final = os.walk(trainfile_path + i).__next__()[0]
            size = len(os.walk(final).__next__()[2])
            if i == "interest":
                num_class0 = size
            elif i == "jobs":
                num_class1 = size
            elif i == "money_supply":
                num_class2 = size
            elif i == "trade":
                num_class3 = size

        #train
        word_counts = self.count_words(training_set)
        self.word_probs = self.word_probabilities(word_counts, num_class0, num_class1, num_class2, num_class3, self.k)

    def classify(self, doc):
        return self.classify_probability(self.word_probs, doc)


model = NaiveBayesClassifier()
model.train(trainfile_path=train_path)

for i in range(20):
    path = test_path+str(i+1)+".txt"
    f = open(path, 'r')
    data = f.read()
    res = model.classify(data)

    max = 0
    idx = 0
    cnt = 0
    for src in res:
        cnt += 1
        if max < src:
            max = src
            idx = cnt

    if idx == 1:
        print("interest")
    elif idx == 2:
        print("jobs")
    elif idx == 3:
        print("money_supply")
    elif idx == 4:
        print("trade")

    f.close()

'''
path2 = "/Users/nam/Desktop/dataset/test/3.txt"
f = open(path2, 'r')
data = f.read()

a = model.classify(data)
print(a)
f.close()
'''

import os

NBC_dev_dir = "./dev/"
NBC_test_dir = "./test/"

class FileOpener:
    def __init__(self):
        self.root_dir = ''
        self.categories = []

    def set_directory(self,directory):
        self.root_dir = directory

    def get_categories(self):
        # 카테고리 불러오기
        categories = os.listdir(self.root_dir)

        # Mac OS용 DS_store 삭제
        if ".DS_Store" in categories:
            categories.remove(".DS_Store")

        self.categories = categories
        return categories

    def get_full_text(self):


    def get_full_text(self, directory, categories):


class NaiveBayesClassifier:

    def load_from_dir(self, directory):
        result = []

        # 카테고리 불러오기
        categories = os.listdir(directory)

        # Mac OS용 DS_store 삭제
        if ".DS_Store" in categories:
            categories.remove(".DS_Store")

        # 카테고리로부터 각각 해당되는 텍스트 불러오기
        for category in categories:
            txt_names = os.listdir(directory + category)
            read_files(directory, category, txt_names)

    def read_files

NBC = NaiveBayesClassifier()
NBC.load_from_dir(NBC_dev_dir)