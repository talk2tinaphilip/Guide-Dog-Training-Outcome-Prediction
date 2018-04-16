import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
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

val data = sc.textFile(“datasets/dataSet5.txt")
data.count()

val parsedData = data.map{line => 
    val parts = line.split(",")
    LabeledPoint(getDoubleValue(parts(5)), Vectors.dense(parts.slice(0,5).map(x => getDoubleValue(x))))
}

val fixedTrainData = sc.makeRDD(parsedData.take((parsedData.count*0.8).toInt))
val fixedTestData = parsedData.subtract(fixedTrainData)



val model2 = new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).run(fixedTrainData)

// Evaluate trained model on training data by forming (label, predicted label) pairs

val labelAndPredsTraining2= fixedTrainData.map { point =>
val prediction = model2.predict(point.features)
  (point.label, prediction)
}

val labelAndPredsTesting2= fixedTestData.map { point =>
val prediction = model2.predict(point.features)
  (point.label, prediction)
}


val trainingError2 = labelAndPredsTraining2.filter(r => r._1 == r._2).count.toDouble / fixedTrainData.count
val testingError2 = labelAndPredsTesting2.filter(r => r._1 == r._2).count.toDouble / fixedTestData.count

printf("\nResult using Fixed split\n")
printf("Prediction reliability on trained data = %.2f%%\n", (100*trainingError2))
printf("Prediction reliability on testing data = %.2f%%\n", (100*testingError2))






