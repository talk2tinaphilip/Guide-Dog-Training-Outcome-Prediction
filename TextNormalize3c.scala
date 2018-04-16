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

val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("playground/3cdataset.csv")

var rowRDD = df.select("dog_SubStatusCode","DayInLife").rdd.map{
case Row(label:String, sentence:String) => (label, textNormalize(sentence))
}.toDF("label", "sentence")

val splits = rowRDD.randomSplit(Array(0.8, 0.2), seed = 11L)
var trainingData = splits(0)
var testData = splits(1)

trainingData = trainingData.withColumn("label", 'label cast DoubleType)

testData = testData.withColumn("label", 'label cast DoubleType)

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("raw")

val remover = new StopWordsRemover().setInputCol("raw").setOutputCol("filtered")

val word2Vec = new Word2Vec().setInputCol("filtered").setOutputCol("features").setVectorSize(3).setMinCount(0)

val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("rawFeatures").setNumFeatures(20)

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.0)

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


