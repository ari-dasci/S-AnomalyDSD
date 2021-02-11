package org.apache.spark.mllib.anomaly

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.ml.linalg.{Vectors, Vector => VectorML}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

/**
 * XGBOD implementation for Apache Spark
 *
 * @param data                 the dataset in Dataset[features, label] format
 * @param n_base_detectors     number of base detectors for the ensemble (default = 10)
 * @param n_selected_detectors number of selected detectors (default = 5)
 * @param strategy             strategy for the selection: random ("rnd"), accuracy ("acc") (default = "rnd")
 * @param threshold            threshold for the accuracy strategy (default = 0.1)
 * @param n_cores              number of cores for XGBoost (default = 360)
 * @param seed                 (default = 48151623)
 * @return an RDD[Double] with the label predicted of each instance in the dataset
 */

class XGBOD_BD(val data: Dataset[Row], val n_base_detectors: Int = 10, val n_selected_detectors: Int = 5, val strategy: String = "rnd", val threshold: Double = 0.1, val n_cores: Int = 360, val seed: Int = 48151623) extends Serializable {

  // Performs the XGBOD algorithm
  def fit(): RDD[Double] = {

    // Transformed Outlier Scores (TOS)
    val outlier_scores: Array[RDD[(Long, Double)]] = Array.fill[RDD[(Long, Double)]](n_base_detectors)(data.sparkSession.sparkContext.emptyRDD)
    if (!data.storageLevel.useMemory) data.persist()

    // Calculate n_base_detectors detectors and join the results
    for (i <- 0 until n_base_detectors) {
      val n_bins_min = 100
      val n_bins_max = 1000
      val rnd = new scala.util.Random
      val n_bins = n_bins_min + rnd.nextInt((n_bins_max - n_bins_min) + 1)
      val scores = new LODA_BD(data, n_bins).fit()
      if (!scores.getStorageLevel.useMemory) scores.persist(StorageLevel.MEMORY_AND_DISK)
      outlier_scores(i) = scores.zipWithIndex().map { case (v, k) => (k, v) }
      outlier_scores(i).persist(StorageLevel.MEMORY_AND_DISK)
    }

    var transformed_outlier_scores = outlier_scores.map(_.mapValues(s => Seq(s))).reduce((a, b) => a.join(b).mapValues { case (s1, s2) => s1 ++ s2 }).sortByKey(ascending = true).map(_._2.toArray)
    transformed_outlier_scores.persist(StorageLevel.MEMORY_AND_DISK)

    // TOS selection
    transformed_outlier_scores = strategy match {
      // Random
      case "rnd" =>
        val tos_list = scala.util.Random.shuffle(0 to n_base_detectors - 1).take(n_selected_detectors)
        transformed_outlier_scores.map { l =>
          var final_tos: Array[Double] = Array[Double]()
          tos_list.foreach { tos =>
            final_tos = final_tos :+ l.apply(tos)
          }
          final_tos
        }
      // Accuracy
      case "acc" =>
        val accuracy = transformed_outlier_scores.map { l =>
          l.map { score =>
            if (score >= threshold) 1.0
            else 0.0
          }
        }
        val labels = data.rdd.map(_.getAs[Double](fieldName = "label"))
        val metrics_results = Array.fill[Double](n_base_detectors)(elem = 0.0)
        for (i <- accuracy.first().indices) {
          val metrics_data = accuracy.map { l => l(i) }.zipWithIndex().map { case (v, k) => (k, v) }.join(labels.zipWithIndex().map { case (v, k) => (k, v) }).map(l => l._2)
          metrics_results(i) = new MulticlassMetrics(metrics_data).accuracy
        }
        val (addSorted, indices) = metrics_results.zipWithIndex.sortWith(_._1 > _._1).unzip
        val indexes = indices.take(n_selected_detectors)
        transformed_outlier_scores.map { l =>
          var final_tos: Array[Double] = Array[Double]()
          indexes.foreach { tos =>
            final_tos = final_tos :+ l.apply(tos)
          }
          final_tos
        }
    }

    import data.sqlContext.implicits._
    val TOS_DF = transformed_outlier_scores.map { l => LabeledPoint(0.0, Vectors.dense(l)) }.toDF().drop(colName = "label").withColumnRenamed(existingName = "features", newName = "scores")

    // Append TOS to data
    val dataIndex: DataFrame = data.sqlContext.createDataFrame(
      data.rdd.zipWithIndex.map(ln => Row.fromSeq(Seq(ln._2) ++ ln._1.toSeq)),
      StructType(Array(StructField("index", LongType, nullable = false)) ++ data.schema.fields)
    )
    val TOS_Index: DataFrame = TOS_DF.sqlContext.createDataFrame(
      TOS_DF.rdd.zipWithIndex.map(ln => Row.fromSeq(Seq(ln._2) ++ ln._1.toSeq)),
      StructType(Array(StructField("index", LongType, nullable = false)) ++ TOS_DF.schema.fields)
    )

    // Join data & scores
    val joined_data = dataIndex.join(TOS_Index, Seq("index")).sort(sortCol = "index").drop(colName = "index")

    val assembler = new VectorAssembler()
      .setInputCols(Array("features", "scores"))
      .setOutputCol("features2")
    val expanded_data = assembler.transform(joined_data).drop(colName = "scores").drop(colName = "features").withColumnRenamed(existingName = "features2", newName = "features")
    expanded_data.persist(StorageLevel.MEMORY_AND_DISK)

    // XGBOD learning
    val xgbParam = Map("eta" -> 0.1f,
      "missing" -> -999,
      "objective" -> "multi:softmax",
      "num_class" -> 2,
      "num_round" -> 50,
      "num_workers" -> n_cores,
      "tree_method" -> "approx")
    val xgbClassifier = new XGBoostClassifier(xgbParam).
      setFeaturesCol("features").
      setLabelCol("label")
    val xgbClassificationModel = xgbClassifier.fit(expanded_data)
    val results = xgbClassificationModel.transform(expanded_data.select(col = "features"))
    results.persist()
    results.count()

    results.rdd.map(_.getAs[Double](fieldName = "prediction"))
  }
}
