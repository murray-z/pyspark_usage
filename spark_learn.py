# -*- coding: utf-8 -*-


from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StringType, StructField, IntegerType
from pyspark.streaming import StreamingContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.stat import Summarizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# 初始化sc
conf = SparkConf().setAppName("learn_spark").setMaster("local[1]")
sc = SparkContext(conf=conf)

# 初始化spark
spark = SparkSession.builder. \
    appName("learn_spark_example"). \
    config("spark.some.config.option", "some-vale"). \
    getOrCreate()


# 采用parallelize初始化RDD
def f1():
    data = [1, 2, 3, 4]
    distData = sc.parallelize(data, numSlices=2)
    sums = distData.reduce(lambda x, y: x + y)
    print(sums)


# 从文件初始化RDD
def f2():
    sc.textFile("data_path")


# 将一个函数传递给sc
def f3():
    def fun(s):
        words = s.split()
        return len(words)

    sc.textFile("file_path").map(fun)


# 错误样例，应该使用accumulator,参考f6()
def f4():
    counter = 0
    rdd = sc.parallelize([1, 2, 3, 4])

    def increment_counter(x):
        global counter
        counter += x

    rdd.foreach(increment_counter)
    print(counter)


# Broadcast Variable
def f5():
    broadcastVar = sc.broadcast([1, 2, 3])
    print(broadcastVar.value)


# accumulator
def f6():
    accum = sc.accumulator(0)
    sc.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
    print(accum.value)


# 从json创建dataframe
def f7():
    df = spark.read.json(people_json_path)
    df.show()
    df.printSchema()
    df.select("name").show()
    df.select(df["name"], df["age"] + 1).show()
    df.filter(df["age"] > 21).show()
    df.groupby("age").count().show()


# SQL, TempView
def f8():
    df = spark.read.json(people_json_path)
    df.createOrReplaceTempView("people")
    sqlDF = spark.sql("select * from people")
    sqlDF.show()


# SQL, GlobalTempView
def f9():
    df = spark.read.json(people_json_path)
    df.createGlobalTempView("people")
    sqlDF = spark.sql("select * from global_temp.people")
    sqlDF.show()


# Inferring the Schema Using Reflection
def f10():
    lines = sc.textFile(people_txt_path)
    parts = lines.map(lambda l: l.split(","))
    people = parts.map(lambda p: Row(name=p[0], age=int(p[1])))

    # infer schema
    schemaPeople = spark.createDataFrame(people)
    schemaPeople.createOrReplaceTempView("people")

    # sql
    teenagers = spark.sql("select * from people where age>=13 and age<=20")
    teenagers.show()

    # df to rdd
    teenNames = teenagers.rdd.map(lambda p: "Name:" + p.name).collect()

    for name in teenNames:
        print(name)


# Programmatically Specifying the Schema
def f11():
    lines = sc.textFile(people_txt_path)
    parts = lines.map(lambda l: l.split(","))
    # to tuple
    people = parts.map(lambda p: (p[0], int(p[1].strip())))
    # Schema String
    schemaString = "name age"

    fields = [StructField(field_name, t(), True)
              for field_name, t in zip(schemaString.split(), [StringType, IntegerType])]
    schema = StructType(fields)

    # apply schema to df
    schemaPeople = spark.createDataFrame(people, schema)
    schemaPeople.show()
    schemaPeople.printSchema()

    # create view
    schemaPeople.createOrReplaceTempView("people")

    # sql
    results = spark.sql("select name from people")
    results.show()


# load and save data
def f12():
    # return df
    df = spark.read.load(user_parquet_path)
    df.select("name", "favorite_color").write.save("nameAndFavoriteColors.parquet")

    # manually specifying options
    df = spark.read.load(people_json_path, format="json")
    df.select("name").write.save("names.parquet", format="parquet")

    # csv
    df = spark.read.load(people_csv_path, format="csv",
                         sep=";", inferSchema="true", header="true")

    # run sql on files directly
    df = spark.sql("select * from parquet.`{}`".format(user_parquet_path))


# spark streaming, 统计一个时间段词频
def f13():
    ssc = StreamingContext(sc, 1)

    # create a Dstream that will connect to hostname:port
    lines = ssc.socketTextStream("hostname", 9999)

    # split each line into words
    words = lines.flatMap(lambda line: line.split(" "))

    # count each word in each batch
    pairs = words.map(lambda word: (word, 1))
    wordCounts = pairs.reduceByKey(lambda x, y: x+y)
    wordCounts.pprint()

    # start the computation
    ssc.start()

    # wait for the computation to terminate
    ssc.awaitTermination()


# Streaming Basic Sources
def f14():
    ssc = StreamingContext(sc, 1)
    # File Streams: 监控路径，例如：hdfs://namenode:8040/logs/
    ssc.textFileStream("data_directory")


# Checkpoint: 保存数据信息
def f15():
    def functionToCreateContext():
        sc = SparkContext()
        ssc = StreamingContext()
        lines = ssc.socketTextStream()
        # 设置checkpoint路径
        ssc.checkpoint("checkpoint_direction")
        return ssc

    # Get StreamingContext from checkpoint data or create a new one
    context = StreamingContext.getOrCreate("checkpoint_direction", functionToCreateContext)

    # start context
    context.start()
    context.awaitTermination()


# MLlib: Correlation
def f16():
    data = [(Vectors.sparse(4, [(0, 1.), (3, -2.)]),),
            (Vectors.dense([4, 5, 0., 3.]),),
            (Vectors.dense([6, 7, 0, 8]),),
            (Vectors.sparse(4, [(0, 9), (3, 1.)]),)]
    df = spark.createDataFrame(data, ["features"])
    df.show()

    r1 = Correlation.corr(df, "features").head().asDict()
    print("Pearson correlation matrix:\n", str(r1[0]))

    r2 = Correlation.corr(df, "features", "spearman").head()
    print("Spearman correlation matrix:\n", str(r2[0]))


# MLlib: ChiSquareTest
def f17():
    data = [(0.0, Vectors.dense(0.5, 10.0)),
            (0.0, Vectors.dense(1.5, 20.0)),
            (1.0, Vectors.dense(1.5, 30.0)),
            (0.0, Vectors.dense(3.5, 30.0)),
            (0.0, Vectors.dense(3.5, 40.0)),
            (1.0, Vectors.dense(3.5, 40.0))]
    df = spark.createDataFrame(data, ["label", "features"])

    r = ChiSquareTest.test(df, "features", "label").head()

    print("pValues: ", str(r.pValues))
    print("degreesOfFreedom: ", str(r.degreesOfFreedom))
    print("statistics: ", str(r.statistics))


# MLlib: Summarizer: max,mean,min,max,std,variance,count
def f18():
    df = sc.parallelize([Row(weight=1.0, features=Vectors.dense(1.0, 1.0, 1.0)),
                         Row(weight=0.0, features=Vectors.dense(1.0, 2.0, 3.0))]).toDF()
    summarizer = Summarizer.metrics("mean", "count")

    # multiple metrics with weight
    df.select(summarizer.summary(df.features, df.weight)).show(truncate=False)
    # multiple metrics without weight
    df.select(summarizer.summary(df.features)).show(truncate=False)
    # single metrics "mean"
    df.select(Summarizer.mean(df.features, df.weight)).show(truncate=False)
    # single metrics "mean" without weight
    df.select(Summarizer.mean(df.features)).show(truncate=False)


# MLlib Pipelines
# main concepts:
# DataFrame: This ML API uses DataFrame from Spark SQL as an ML dataset,
# which can hold a variety of data types. E.g., a DataFrame could have different columns storing text,
# feature vectors, true labels, and predictions.

# Transformer: A Transformer is an algorithm which can transform one DataFrame into another DataFrame.
# E.g., an ML model is a Transformer which transforms a DataFrame with features into a DataFrame with predictions.

# Estimator: An Estimator is an algorithm which can be fit on a DataFrame to produce a Transformer.
# E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model.

# Pipeline: A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow.

# Parameter: All Transformers and Estimators now share a common API for specifying parameters.


# MLlib: Estimator, Transformer, and Param
def f19():
    # training data
    training = spark.createDataFrame([
        (1.0, Vectors.dense([0.0, 1.1, 0.1])),
        (0.0, Vectors.dense([2.0, 1.0, -1.0])),
        (0.0, Vectors.dense([2.0, 1.3, 1.0])),
        (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])

    # create lr instance, this is an Estimator
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    # print out the parameters, documentation, and the default values
    print("lr param: ", lr.explainParam())

    # train model
    model1 = lr.fit(training)

    # model1 is a transformer
    print("Model1 was fit using parameters:")
    print(model1.extractParamMap())

    # specify parameters
    paramMap = {lr.maxIter: 20}
    paramMap[lr.maxIter] = 30
    paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55})

    # change output col name
    paramMap2 = {lr.predictionCol: "myProbCol"}
    paramMapCombined = paramMap.copy()
    paramMapCombined.update(paramMap2)

    # learn new model
    model2 = lr.fit(training, paramMapCombined)
    print("model2 was fit use params:")
    print(model2.extractParamMap())

    # Prepare test data
    test = spark.createDataFrame([
        (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
        (0.0, Vectors.dense([3.0, 2.0, -0.1])),
        (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

    prediction = model2.transform(test)
    result = prediction.select("features", "label", "myProbCol", "prediction").collect()

    for row in result:
        print(row.features, row.label, row.myProbCol, row.prediction)


# Example Pipeline
def f20():
    # Prepare training documents from a list of (id, text, label) tuples.
    training = spark.createDataFrame([
        (0, "a b c d e spark", 1.0),
        (1, "b d", 0.0),
        (2, "spark f g h", 1.0),
        (3, "hadoop mapreduce", 0.0)
    ], ["id", "text", "label"])

    # pipeline with three stage: tokenizer, hashingTF, lr
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getInputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # training
    model = pipeline.fit(training)

    # Prepare test documents, which are unlabeled (id, text) tuples.
    test = spark.createDataFrame([
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "spark hadoop spark"),
        (7, "apache hadoop")
    ], ["id", "text"])

    # predict
    prediction = model.transform(test)
    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        print(row.id, row.text, row.probability, row.prediction)


# Binomial logistic regression
def f21():
    # load training data
    training = spark.read.format("libsvm").load(libsvm_data_path)

    # lr instance
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # fit model
    lrModel = lr.fit(training)

    # print coefficients and intercept
    print("coefficient:", lrModel.coefficients)
    print("intercept:", lrModel.intercept)

    # use multinomial family for binary classification
    mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

    # fit model
    mlrModel = mlr.fit(training)

    print("coefficient:", mlrModel.coefficientMatrix)
    print("intercept:", mlrModel.interceptVector)

    # Summary
    trainingSummary = lrModel.summary

    # ROC and AUC
    trainingSummary.roc.show()
    print("areaUnderROC:", trainingSummary.areaUnderROC)

    # set the model threshold to max F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max("F-Measure").select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']).\
        select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)


# Decision Tree classifier
def f22():
    # load data
    data = spark.read.format("libsvm").load(libsvm_data_path)

    # Index labels, transform label to index
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # automatically identify categorical features, and index them
    # features with > 4 distinct values are treated as continuous
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # split data
    trainingData, testData = data.randomSplit([0.7, 0.3])

    # init dt
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # train model
    model = pipeline.fit(trainingData)

    # predict
    predictions = model.transform(testData)

    # show some data
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # evaluate
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error:", 1. - accuracy)

    # get tree model
    treeModel = model.stages[2]

    # summary
    print(treeModel)

if __name__ == '__main__':
    people_json_path = "/usr/local/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/resources/people.json"
    people_txt_path = "/usr/local/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/resources/people.txt"
    people_csv_path = "/usr/local/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/resources/people.csv"
    user_parquet_path = "/usr/local/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/resources/users.parquet"
    libsvm_data_path = "/usr/local/spark/spark-3.2.0-bin-hadoop3.2/data/mllib/sample_libsvm_data.txt"
    f21()
