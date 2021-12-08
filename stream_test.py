from pyspark.context import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
from sklearn.feature_extraction import text

from joblib import load

import matplotlib.pyplot as plt

import json
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Streams a file to a Spark Streaming Context')
parser.add_argument('--batch-size', '-b', help='Batch size',
                    required=False, type=int, default=100)

# Initialize the spark context.
sc = SparkContext(appName="SpamStreaming")
ssc = StreamingContext(sc, 5)

spark = SparkSession(sc)

schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True), StructField("feature2", StringType(), True)])

vectorizer = HashingVectorizer(alternate_sign=False)
le = LabelEncoder()
mnb = MultinomialNB()
sgd = SGDClassifier(warm_start=True)
per = Perceptron(warm_start=True)
kmeans = MiniBatchKMeans(n_clusters=2)

args = parser.parse_args()

count = 0
TEST_SIZE = int(3373/args.batch_size)

acc = [[] for i in range(4)]
pre = [[] for i in range(4)]
rec = [[] for i in range(4)]
f1 = [[] for i in range(4)]

mnb = load('mnb' + str(args.batch_size) + '.pkl')
per = load('per' + str(args.batch_size) + '.pkl')
sgd = load('sgd' + str(args.batch_size) + '.pkl')
kmeans = load('kmeans' + str(args.batch_size) + '.pkl')

def removeNonAlphabets(s):
    s.lower()
    regex = re.compile('[^a-z\s]')
    s = regex.sub('', s)   
    return s

def removeStopWords(s):
    stop_words = list(text.ENGLISH_STOP_WORDS)
    res = []

    for sentence in s:
        words = sentence.split()
        temp = []
        for word in words:
            if word not in stop_words:
                temp.append(word)
        
        temp = ' '.join(temp)
        res.append(temp)
    
    return res

def print_stats(index, y, pred):
    acc[index].append(accuracy_score(y, pred))
    pre[index].append(precision_score(y, pred))
    rec[index].append(recall_score(y, pred))
    conf_m = confusion_matrix(y, pred)
    f1[index].append(f1_score(y, pred))

    # print(f"\naccuracy: %.3f" %acc[index][-1])
    # print(f"precision: %.3f" %pre[index][-1])
    # print(f"recall: %.3f" %rec[index][-1])
    # print(f"f1-score : %.3f" %f1[index][-1])
    print(f"confusion matrix: ")
    print(conf_m)

    print(classification_report(y, pred, labels = [0, 1]))

def plotting(arr, str):
    x_axis = [i for i in range(1, TEST_SIZE + 1)]
    plt.plot(x_axis, arr[0], label='MultinomialNB')  
    plt.plot(x_axis, arr[1], label='Perceptron') 
    plt.plot(x_axis, arr[2], label='SGD-classifier') 
    plt.plot(x_axis, arr[3], label='K-means') 
    plt.ylabel(str)     
    plt.xlabel("Num Of Batches")    
    plt.title(str) 
    plt.legend()
    plt.show()

def func(rdd):
    global count, TEST_SIZE
    l = rdd.collect()

    if len(l):  
        count += 1

        df = spark.createDataFrame(json.loads(l[0]).values(), schema)

        df_list = df.collect()
        
        # Remove non alphabetic characters
        non_alphabetic = [(removeNonAlphabets(x['feature0'] + ' ' + x['feature1'])) for x in df_list]

        # Remove stop words
        no_stop_words = removeStopWords(non_alphabetic)

        X_test = vectorizer.fit_transform(no_stop_words)

        y_test = le.fit_transform(np.array([x['feature2']  for x in df_list]))

        #multinomial nb
        pred = mnb.predict(X_test)
        print("\nMultinomial NB: ")
        print_stats(0, y_test, pred)

        #perceptron
        pred = per.predict(X_test)
        print("\nPerceptron: ")
        print_stats(1, y_test, pred)

        #sgdclassifier
        pred = sgd.predict(X_test)
        print("\nSGD Classifier: ")
        print_stats(2, y_test, pred)

        #k means clustering
        pred = kmeans.predict(X_test)
        print("\nK-Means: ")
        print_stats(3, y_test, pred)
    
    if count == TEST_SIZE:
        plotting(acc, "Accuracy")
        plotting(pre, "Precision")
        plotting(rec, "Recall")
        plotting(f1, "F1")
        count = 0


lines = ssc.socketTextStream("localhost", 6100)

lines.foreachRDD(func)

ssc.start()
ssc.awaitTermination()
ssc.stop()