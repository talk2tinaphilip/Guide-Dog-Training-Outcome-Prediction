import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}

def getDoubleValue(input: String): Double = input match {
 case "0" => 0.0
 case "1" => 1.0
 case "2" => 2.0
 case "3" => 3.0
 case "4" => 4.0
 case "5" => 5.0
 case "6" => 6.0
 case "7" => 7.0
}

val data = sc.textFile(“datasets/2fdataset.txt")
data.count()


val parsedData = data.map{line => 
    val parts = line.split(",")
    LabeledPoint(getDoubleValue(parts(5)), Vectors.dense(parts.slice(0,5).map(x => getDoubleValue(x))))
}

val splits = parsedData.randomSplit(Array(0.8, 0.2), seed = 11L)
val trainingData = splits(0)
val testData = splits(1)


val model1 = new LogisticRegressionWithSGD().setIntercept(true).run(trainingData)

// Evaluate trained model on training data by forming (label, predicted label) pairs
val labelAndPredsTraining= trainingData.map { point =>
val prediction = model1.predict(point.features)
  (point.label, prediction)
}

// Evaluate trained model on training data by forming (label, predicted label) pairs
val labelAndPredsTesting= testData.map { point =>
val prediction = model1.predict(point.features)
  (point.label, prediction)
}

//find the ratio of correct predciton to that of the entire dataset
val trainingError = labelAndPredsTraining.filter(r => r._1 == r._2).count.toDouble / trainingData.count
val testingError = labelAndPredsTesting.filter(r => r._1 == r._2).count.toDouble / testData.count


printf("Result using Random Split and SGD\n")
printf("Prediction reliability on trained data = %.2f%%\n", (100*trainingError))
printf("Prediction reliability on testing data = %.2f%%\n", (100*testingError))





