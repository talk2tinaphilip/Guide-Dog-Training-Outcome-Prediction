import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import java.text.Normalizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import java.io._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.Word2Vec


val data = sc.textFile("datasets/3adataset.txt")
data.count()


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

val parsedData = data.map{line => 
var parts = line.split(',')
textNormalize(parts(1))
}
val sparkSession =  SparkSession.builder().getOrCreate()
val parsedDataSet =  sparkSession.createDataset(parsedData).toDF("sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val wordsData = tokenizer.transform(parsedDataSet).toDF("id", "raw")

val remover = new StopWordsRemover().setInputCol("raw").setOutputCol("filtered")

var finaldata = remover.transform(wordsData)

val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("rawFeatures").setNumFeatures(20)

val featurizedData = hashingTF.transform(finaldata)

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)
val rescaledData = idfModel.transform(featurizedData)

println("\nOutput after TF-IDF\n")
rescaledData.select("id", "features").show()

val word2Vec = new Word2Vec().setInputCol("filtered").setOutputCol("features").setVectorSize(20).setMinCount(0)
val model = word2Vec.fit(finaldata)
val wordToVecRes = model.transform(finaldata)

println("\nOutput after Word2Vec\n")
wordToVecRes.select("id", "features").show()


