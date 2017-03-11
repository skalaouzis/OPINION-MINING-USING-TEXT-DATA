import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.sql._
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object MySVM {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("SVMWithSGD")
    val sc = new SparkContext(conf)
    
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val currentDir = System.getProperty("user.dir")  // get the current directory
    val positives = "file://" + currentDir + "/compinestopped.txt"  // define the positives file
    val negatives = "file://" + currentDir + "/compinebadstopped.txt" // define the negatives file
    
    //set positives to (1, review)
    val positivesRDD = sc.textFile(positives).map(review => review.trim()).map(review => review.replaceAll("\\s{2,}", " ")).map(review => (1D, review))

    //set negatives to (0, review)
    val negativesRDD = sc.textFile(negatives).map(review => review.trim()).map(review => review.replaceAll("\\s{2,}", " ")).map(review => (0D, review))

    //positives and negatives to one RDD with unique id
    val unionRDD = positivesRDD.union(negativesRDD).zipWithUniqueId().map(review => (review._2, review._1._2, review._1._1))
    //set dataframe column names
    val training = unionRDD.toDF("id", "review", "label")

    val tokenizer = new Tokenizer()
      .setInputCol("review")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    //val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")

    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF))
    val pipelineModel = pipeline.fit(training)
    pipelineModel.save("bdc/scalapipelinemodel") // Saving the pipeline model

    // transforming the training DataFrame with the pipeline model and then converting it to a dataset or dataframe of LabeledPoints
    val transformed_training = pipelineModel.transform(training)
      //.select("features", "label").map( row => LabeledPoint(row.getAs[Double]("label"), SparseVector.fromML(row.getAs[org.apache.spark.ml.linalg.Vector]("features"))))
    // splitting the RDD of LabeledPoints (the whole training set). SVM will be trained on training_set and validated to the other 40% of the initial set
    val splits = transformed_training.randomSplit(Array(0.65, 0.35), seed = 11L)
    val training_set = splits(0).cache()
    val validation_set = splits(1)
    
    // run the SVM algorithm with the configured parameters on an input RDD of LabeledPoint entries
    val numIterations = 100
    val svm_model = new SVMWithSGD().train(training_set, numIterations)
    svm_model.clearThreshold()
    //svm_model.save(sc,”somewhere”) // Saving SVM model for prediction

    val scoreAndLabels = validation_set.map { point =>
      val score = svm_model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    // Save and load model
    svm_model.save(sc, "bdc/scalaSVMWithSGDModel")
    //val sameModel = SVMModel.load(sc, "bdc/scalaSVMWithSGDModel")

  }
}
