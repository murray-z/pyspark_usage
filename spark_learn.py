# -*- coding: utf-8 -*-


from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StringType, StructField, IntegerType


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



if __name__ == '__main__':
    people_json_path = "/usr/local/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/resources/people.json"
    people_txt_path = "/usr/local/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/resources/people.txt"
    people_csv_path = "/usr/local/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/resources/people.csv"
    user_parquet_path = "/usr/local/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/resources/users.parquet"
    f12()
