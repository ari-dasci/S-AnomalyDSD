package org.apache.spark.mllib.anomaly

import breeze.linalg._
import breeze.stats._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vectors, Vector => VectorML}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

import scala.math.sqrt

/**
 * LSCP implementation for Apache Spark
 *
 * @param data             the dataset in Dataset[Row] format
 * @param n_base_detectors number of base detectors for the ensemble (default = 10)
 * @param strategy         strategy for the pseudo ground truth generation: average ("avg"), maximum ("max") (default = "max")
 * @param clus_method      clustering method for the local neigborhood calculation: "kmeans", "bisec" (default = "kmeans")
 * @param n_clus           number of clusters for the local neigborhood calculation (default = 11)
 * @param dcs              percentage of base detectors selected for dynamic outlier ensemble selection
 * @return an RDD[Double] with the scores of each instance of the dataset
 */

class LSCP_BD(val data: Dataset[Row], val n_base_detectors: Int = 10, val strategy: String = "avg", val clus_method: String = "kmeans", val n_clus: Int = 11, val dcs: Double = 0.5) extends Serializable {

  /**
   * Calculates de Pearson Correlation between two vectors
   */

  def pearson(a: Vector[Double], b: Vector[Double]): Double = {
    if (a.length != b.length)
      throw new IllegalArgumentException("Vectors not of the same length.")

    val n = a.length

    val dot = a.dot(b)
    val adot = a.dot(a)
    val bdot = b.dot(b)
    val amean = mean(a)
    val bmean = mean(b)

    (dot - n * amean * bmean) / (sqrt(adot - n * amean * amean) * sqrt(bdot - n * bmean * bmean))
  }

  // Performs the LSCP algorithm
  def fit(): RDD[Double] = {

    // Base Detector Generation
    if (!data.storageLevel.useMemory) data.persist(StorageLevel.MEMORY_AND_DISK)
    val base_detector_scores: Array[RDD[(Long, Double)]] = Array.fill[RDD[(Long, Double)]](n_base_detectors)(data.sparkSession.sparkContext.emptyRDD)

    for (i <- 0 until n_base_detectors) {
      val n_bins_min = 100
      val n_bins_max = 1000
      val rnd = new scala.util.Random
      val n_bins = n_bins_min + rnd.nextInt((n_bins_max - n_bins_min) + 1)
      val scores = new LODA_BD(data, n_bins).fit()
      if (!scores.getStorageLevel.useMemory) scores.persist(StorageLevel.MEMORY_AND_DISK)
      base_detector_scores(i) = scores.zipWithIndex().map { case (v, k) => (k, v) }
      base_detector_scores(i).persist(StorageLevel.MEMORY_AND_DISK)
    }

    var pseudo_ground_truth = base_detector_scores.map(_.mapValues(s => Seq(s))).reduce((a, b) => a.join(b).mapValues { case (s1, s2) => s1 ++ s2 }).sortByKey(ascending = true).map(_._2.toArray)
    pseudo_ground_truth.persist(StorageLevel.MEMORY_AND_DISK)

    // Pseudo Ground Truth Generation
    pseudo_ground_truth = strategy match {
      case "avg" => pseudo_ground_truth.map { l => l :+ l.sum / l.length }
      case "max" => pseudo_ground_truth.map { l => l :+ l.max }
    }

    // Local Region Definition
    import data.sqlContext.implicits._
    val pseudo_gt_DF = pseudo_ground_truth.map { l => LabeledPoint(0.0, Vectors.dense(l)) }.toDF().drop(colName = "label").withColumnRenamed(existingName = "features", newName = "scores")

    // Add index to each instance
    val dataIndex: DataFrame = data.sqlContext.createDataFrame(
      data.rdd.zipWithIndex.map(ln => Row.fromSeq(Seq(ln._2) ++ ln._1.toSeq)),
      StructType(Array(StructField("index", LongType, nullable = false)) ++ data.schema.fields)
    )
    val gt_Index: DataFrame = pseudo_gt_DF.sqlContext.createDataFrame(
      pseudo_gt_DF.rdd.zipWithIndex.map(ln => Row.fromSeq(Seq(ln._2) ++ ln._1.toSeq)),
      StructType(Array(StructField("index", LongType, nullable = false)) ++ pseudo_gt_DF.schema.fields)
    )

    // Join data & scores
    val data_gt = dataIndex.join(gt_Index, Seq("index"))
    data_gt.persist(StorageLevel.MEMORY_AND_DISK)

    // Cluster Partitioning
    val gtSortedByRegion = new Cluster_Partitioning(data = data_gt, clus_method, n_clus, max_size = 10000, min_size = 1000).balance_clusters()
    gtSortedByRegion.persist(StorageLevel.MEMORY_AND_DISK)

    // Model Selection and Combination
    val gtSortedRDD = gtSortedByRegion.select(col = "index", cols = "scores").rdd.map { row =>
      val index = row.getAs[Long](fieldName = "index")
      val featuresML = row.getAs[VectorML](fieldName = "scores")
      val features = org.apache.spark.mllib.linalg.Vectors.fromML(featuresML).toArray
      (index, features)
    }
    gtSortedRDD.persist(StorageLevel.MEMORY_AND_DISK)

    // Correlation Calculation
    val scores = gtSortedRDD.mapPartitions { partition =>
      val data = partition.toArray
      if (data.length > 0) {
        val transposed = data.map(_._2).transpose
        val gt = transposed.last
        val correlations = Array.fill[Double](n_base_detectors)(elem = 0.0)
        for (i <- 0 until n_base_detectors) {
          correlations(i) = Math.abs(pearson(breeze.linalg.Vector(transposed(i)), breeze.linalg.Vector(gt)))
        }
        // MAX / MOA / AOM
        if (dcs == 1.0) {
          val index = correlations.indexOf(correlations.max)
          data.map{l => (l._1, l._2.apply(index))}.toIterator
        } else {
          var num_pgt = (n_base_detectors * dcs).toInt
          if (num_pgt < 1) num_pgt = 1
          val sorted = correlations.sortWith(_ > _)
          val mostCorrelated = Array.fill[Int](num_pgt)(elem = -1)
          for (j <- 0 until num_pgt) {
            mostCorrelated(j) = correlations.indexOf(sorted(j))
          }
          if (strategy.equals("avg")) {
            data.map { l =>
              val values = Array.fill[Double](num_pgt)(elem = Double.NegativeInfinity)
              for (j <- 0 until num_pgt) {
                values(j) = l._2(mostCorrelated(j))
              }
              (l._1, values.max)
            }.toIterator
          } else {
            data.map { l =>
              val values = Array.fill[Double](num_pgt)(elem = Double.NegativeInfinity)
              for (j <- 0 until num_pgt) {
                values(j) = l._2(mostCorrelated(j))
              }
              (l._1, values.sum / values.length)
            }.toIterator
          }
        }
      } else {
        Array[(Long, Double)]().toIterator
      }
    }
    scores.sortByKey(ascending = true).map(_._2)
  }
}
