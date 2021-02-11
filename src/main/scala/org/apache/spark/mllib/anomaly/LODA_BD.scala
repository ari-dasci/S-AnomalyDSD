package org.apache.spark.mllib.anomaly

import org.apache.spark.ml.linalg.{Vector => VectorML}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrices, Matrix}
import org.apache.spark.mllib.random.StandardNormalGenerator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel

/**
 * LODA implementation for Apache Spark
 *
 * @param data          the dataset in Dataset[Row] format
 * @param n_bins        number of bins for the histograms (default = 10)
 * @param n_random_cuts number of projections (default = 100)
 * @param seed          (default = 48151623)
 * @return an RDD[Double] with the scores of each instance of the dataset
 */

class LODA_BD(val data: Dataset[Row], val n_bins: Int = 100, val n_random_cuts: Int = 100, val seed: Long = 48151623) extends Serializable {

  private val dataAsVector: RDD[VectorML] = data.select(col = "features").rdd.map { case Row(v: VectorML) => v }
  private val dataAsMatrix = new RowMatrix(dataAsVector.map(org.apache.spark.mllib.linalg.Vectors.fromML))
  private val n_components: Int = dataAsVector.first.size
  private val n_nonzero_components: Int = Math.sqrt(n_components).toInt
  private val weights: Array[Double] = Array.fill(n_random_cuts)(1.0).map(_ / n_random_cuts)

  /**
   * Creates an Array of size n_components x n_random_cuts for the projection of the data:
   * It generates Arrays of size n_components filled with 0.0 values
   * Sqrt(n_components) random components are selected
   * Those selected values are changed with the values drawn from a normal distribution
   */
  private def createRandomArray(): Array[Double] = {
    val generator: StandardNormalGenerator = new StandardNormalGenerator()
    generator.setSeed(seed)
    var gaussianData = Array[Double]()
    val r = scala.util.Random
    r.setSeed(seed)

    for (i <- 0 until n_random_cuts) {
      val features = Array.fill(n_nonzero_components)(r.nextInt(n_components))
      val arrayZeros = Array.fill(n_components)(0.0)

      for (j <- features.indices) {
        arrayZeros(features(j)) = generator.nextValue()
      }
      gaussianData = gaussianData ++ arrayZeros
    }
    gaussianData
  }

  // Performs the LODA algorithm
  def fit(): RDD[Double] = {

    // Projection calculation
    val projections: Matrix = Matrices.dense(n_random_cuts, n_components, createRandomArray())
    val projectedMatrix: RowMatrix = dataAsMatrix.multiply(projections.transpose)
    val projectedData: RDD[linalg.Vector] = projectedMatrix.rows
    projectedData.persist(StorageLevel.MEMORY_AND_DISK)

    // Histograms initialization
    val limits: Array[Array[Double]] = Array.fill(n_random_cuts, n_bins + 1)(0.0)
    val histograms: Array[Array[Double]] = Array.fill(n_random_cuts, n_bins)(0)

    // Histogram calculation and normalization
    for (i <- 0 until n_random_cuts) {
      val histogram = projectedData.map(v => v(i)).histogram(n_bins)
      limits(i) = histogram._1
      histograms(i) = histogram._2.map(_ + 1e-12)
      val histogram_sum = histograms(i).sum
      histograms(i) = histograms(i).map(_ / histogram_sum)
    }

    // Scores calculation based on drawn histograms
    val pred_scores: RDD[Double] = projectedData.map { l =>
      val values = l.toArray
      val scores = Array.fill(n_random_cuts)(Double.NegativeInfinity)
      for (i <- 0 until n_random_cuts) {
        val index = (limits(i).drop(1).dropRight(1) :+ values(i)).sorted.indexOf(values(i))
        scores(i) = -weights(i) * Math.log(histograms(i)(index))
      }

      // Scores normalization
      scores.sum / n_random_cuts
    }

    projectedData.unpersist()
    pred_scores
  }
}
