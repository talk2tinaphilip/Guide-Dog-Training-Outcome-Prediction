import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import java.text.Normalizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.Word2Vec
import java.io._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType};

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._

def textNormalize(input: String): String = {
  
  // Text Normalization
  var inputNormalized = Normalizer.normalize(input, Normalizer.Form.NFD)
 
  // Remove Diacritical Marks
  inputNormalized = inputNormalized.replaceAll("[^\\p{ASCII}]", "")

  // Replace punctuation marks with space
  inputNormalized = inputNormalized.replaceAll("[^-a-zA-Z0-9]", " ")

  // Remove extra whitespaces
  inputNormalized = inputNormalized.replaceAll("\\s+", " ")

  // Covert all characters to lower case
  inputNormalized.toLowerCase()

} 

val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("3c/datasets/3cdataset.csv")
var rowRDD = df.select("dog_SubStatusCode","DayInLife").rdd

val trainingRowRDD = sc.makeRDD(rowRDD.take((rowRDD.count * 0.8).toInt))
print("------------log----------------\n# of training data: " + trainingRowRDD.count + "\n------------log----------------\n")
val testingRowRDD = rowRDD.subtract(trainingRowRDD)
print("------------log----------------\n# of testing data: " + testingRowRDD.count + "\n------------log----------------\n")

// The schema is encoded in a string
val schemaString = "label sentence"
// Generate the schema based on the string of schema
val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, nullable = true))
val schema = StructType(fields)


val trainingData = spark.createDataFrame(trainingRowRDD, schema).withColumn("label", 'label.cast("Double"))
val testData = spark.createDataFrame(testingRowRDD, schema).withColumn("label", 'label.cast("Double"))

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("raw")

val remover = new StopWordsRemover().setInputCol("raw").setOutputCol("filtered")

val word2Vec = new Word2Vec().setInputCol("filtered").setOutputCol("features").setVectorSize(100).setMinCount(0)

val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("rawFeatures").setNumFeatures(262144)

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

val lr = new LogisticRegression().setMaxIter(200).setRegParam(0.9)

val pipeline2 = new Pipeline().setStages(Array (tokenizer, remover, word2Vec, lr))

val pipeline1 = new Pipeline().setStages(Array (tokenizer, remover, hashingTF, idf, lr))

val model1 = pipeline1.fit(trainingData) 

val trainoutput1 = model1.transform(trainingData)
val testoutput1 = model1.transform(testData)

var trainingError1 = trainoutput1.filter(r => r(0) == r(8)).count.toDouble / trainingData.count
var testingError1 = testoutput1.filter(r => r(0) == r(8)).count.toDouble / testData.count

val model2 = pipeline2.fit(trainingData) 

val trainoutput2 = model2.transform(trainingData)
val testoutput2 = model2.transform(testData)

var trainingError2 = trainoutput2.filter(r => r(0) == r(7)).count.toDouble / trainingData.count
var testingError2 = testoutput2.filter(r => r(0) == r(7)).count.toDouble / testData.count


printf("\nResult using TF-IDF and Random Split\n")
printf("Prediction reliability on trained data = %.2f%%\n", (100*trainingError1))
printf("Prediction reliability on testing data = %.2f%%\n", (100*testingError1))

printf("\nResult using word2Vec and Random Split\n")
printf("Prediction reliability on trained data = %.2f%%\n", (100*trainingError2))
printf("Prediction reliability on testing data = %.2f%%\n", (100*testingError2))


