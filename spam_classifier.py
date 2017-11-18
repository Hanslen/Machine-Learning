#coding=utf-8

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora, models
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
import re
import os
import time


train_spam_path = "/Users/Hanslen/Desktop/Machine Learning/Hw3/email_classification_data/train_data/spam/"
train_ham_path = "/Users/Hanslen/Desktop/Machine Learning/Hw3/email_classification_data/train_data/ham/"
test_path = "/Users/Hanslen/Desktop/Machine Learning/Hw3/email_classification_data/test_data/"
target_path = "/Users/Hanslen/Desktop/Machine Learning/Hw3/email_classification_data/combine.txt"

y_label = np.array([1]*1265 + [0]*3107)


def text_filter(path1, path2):
    f = open(path2, 'a')
    for (dirpath, dirnames, filenames) in os.walk(path1):
        filenames = filenames[1:]
        filenames = sorted(filenames,  key=lambda x: int(re.findall("\d+", x)[0]))
        for filename in filenames:
            if '.txt' in filename:
                content = preprocessing(path1+filename)
                f.write(content)
                f.write("\n")
    f.close()


def preprocessing(path):
    stemmer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    f = open(path)
    result = []
    for line in f.readlines():
        words = re.findall("[a-zA-Z]+", line)
        for word in words:
            if word not in stop_words and len(word) > 1:
                result.append(stemmer.lemmatize(word.lower(), "v"))
    return " ".join(result)


def bag_of_words(path):
    t_start = time.time()
    print 'loading text'
    f = open(path)  # LDA_test.txt
    texts = [[word for word in line.strip().lower().split()] for line in f]
    # texts = [line.strip().split() for line in f]
    print 'loaded, using time %.3f s' % (time.time() - t_start)
    f.close()
    M = len(texts)
    print 'number of emails : %d' % M
    # pprint(texts)
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    print "vocabulary size: %d" % V
    corpus = [dictionary.doc2bow(text) for text in texts]
    print 'Calculating TF-IDF --'
    t_start = time.time()
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    print 'Calculating TF-IDF finished, using time %.3f s' % (time.time() - t_start)
    return corpus_tfidf, dictionary, V


def tfidf_to_array(tfidf, voc_size):
    tfidf_array = []
    for article in tfidf:
        article_array = [0]*voc_size
        for word in article:
            article_array[word[0]-1] = word[1]
        tfidf_array.append(article_array)
    return np.array(tfidf_array)


def reduce_dim(tfidf_array, dimension=5000):
    start_time = time.time()
    print "start reducing dimension --"
    pca = PCA(dimension)
    pca.fit(tfidf_array)
    print "finish reducing dimension, using time: %.3f " %(time.time() - start_time)
    return pca.transform(tfidf_array)


def reduce_dim2(corpus_tfidf, num_topics, dictionary):
    lsi = models.LsiModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary)
    topic_result = [a for a in lsi[corpus_tfidf]]
    result = []
    for article in topic_result:
        article_result = [0] * num_topics
        for i in range(len(article)):
            article_result[i] = article[i][1]
        result.append(article_result)
    return np.array(result)


def SVM_classifier(x, y, C):
    x_train = x[:4372]
    x_test = x[4372:]
    y_train = y
    model = svm.SVC(C=C, kernel='rbf', gamma=0.001)
    model.fit(x_train, y_train)
    print "training_accuracy:", np.mean(y_train == model.predict(x_train))
    return model.predict(x_test)


if __name__ == '__main__':
    tfidf, dictionary, voc_size = bag_of_words(target_path)
    tfidf_array = tfidf_to_array(tfidf, voc_size)
    array1 = reduce_dim(tfidf_array, 3000)
    array2 = reduce_dim2(tfidf_array, 3000, dictionary)
    reduced_array = np.concatenate((array1, array2), axis=1)
    result = SVM_classifier(reduced_array, y_label, 2000)
