# importing required libraries

import json
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext

from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.sql.functions import length
from pyspark.ml import Pipeline

# getting the sparkContext - which creates a new spark job

sc = SparkContext("local[2]", "DetectionOfSpam")
ssc = StreamingContext(sc, 1)
spark = SparkSession.builder.getOrCreate()

'''
def tokenizer_func(df):
	tokenizer = Tokenizer(inputCol="feature1", outputCol="message_token")
	countTokens = udf(lambda message_token: len(message_token), IntegerType())
	tokenized = tokenizer.transform(df)
	tokenized.select("feature1", "message_token")\
	.withColumn("tokens", countTokens(col("message_token"))).show(truncate=False)
'''
# Preprocessing Function

def dataclean(df):
	df = df.withColumn('length',length(df['feature1']))
	tokenizer = Tokenizer(inputCol="feature1", outputCol="token_text")
	stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
	count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
	idf = IDF(inputCol="c_vec", outputCol="tf_idf")
	ham_spam_to_num = StringIndexer(inputCol='feature2',outputCol='label')
	clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
	data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
	cleaner = data_prep_pipe.fit(df)
	clean_data = cleaner.transform(df)
	clean_data.show()
	
# Function to read the data stream and print the respective data read.

def RDDtoDf(x):
    if not x.isEmpty():
        y = x.collect()[0]
        z = json.loads(y)
        df=spark.createDataFrame(z.values())
        #tokenizer_func(df)
        dataclean(df)
        #df.show()
        #print(k)

# print(type(rdd),type(df))

records = ssc.socketTextStream("localhost", 6100)
records.foreachRDD(RDDtoDf)

ssc.start()
ssc.awaitTermination()
ssc.stop()