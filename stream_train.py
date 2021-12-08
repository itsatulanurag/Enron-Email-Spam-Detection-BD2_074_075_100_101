from pyspark.context import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import text

from joblib import dump

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

def func(rdd):

    l = rdd.collect()

    if len(l):
        df = spark.createDataFrame(json.loads(l[0]).values(), schema)

        df_list = df.collect()
        
        # Remove non alphabetic characters
        non_alphabetic = [(removeNonAlphabets(x['feature0'] + ' ' + x['feature1'])) for x in df_list]

        # Remove stop words
        no_stop_words = removeStopWords(non_alphabetic)

        X_train = vectorizer.fit_transform(no_stop_words)

        y_train = le.fit_transform(np.array([x['feature2']  for x in df_list]))

        #multinomial nb
        mnb.partial_fit(X_train, y_train, classes = np.unique(y_train))

        #perceptron
        per.partial_fit(X_train, y_train, classes = np.unique(y_train))

        #sgdclassifier
        sgd.partial_fit(X_train, y_train, classes = np.unique(y_train))

        #k means clustering
        kmeans.partial_fit(X_train, y_train)

        dump(mnb, 'mnb' + str(args.batch_size) + '.pkl', compress=9)
        dump(per, 'per' + str(args.batch_size) + '.pkl', compress=9)
        dump(sgd, 'sgd' + str(args.batch_size) + '.pkl', compress=9)
        dump(kmeans, 'kmeans' + str(args.batch_size) + '.pkl', compress=9)


lines = ssc.socketTextStream("localhost", 6100)

lines.foreachRDD(func)

ssc.start()
ssc.awaitTermination()
ssc.stop()