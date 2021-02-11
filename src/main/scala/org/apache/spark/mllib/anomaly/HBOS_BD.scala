package org.apache.spark.mllib.anomaly

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

/**
 * HBOS algorithm
 *
 * @param dataset  the complete dataset
 * @param n_bins   the number of bins
 * @param strategy "static" or "dynamic", choose the version
 * @return a dataset with two columns, unix and scores
 */

class HBOS_BD(dataset: Dataset[Row], n_bins: Int = 100, strategy: String = "static") extends Serializable {
  /**
   * computes the score of each instance for the static version
   *
   * @param iter      the instances
   * @param histogram the histograms
   * @param limits    the upper and lower bound of the histograms
   * @param k         the number of bins
   * @return the index of each instance and his score
   */
  private def scoresMapStatic(iter: Iterator[(Row, Long)], histogram: Broadcast[Array[Array[Double]]], limits: Broadcast[Array[Array[Double]]], k: Int): Iterator[Array[(Long, Double)]] = {
    // index to compute the bin of each attribute
    var index: Int = 0

    // number of instances in each partition
    var length: Int = 0

    // auxiliar iterator to compute the number of instances in each partition
    val newIterator: (Iterator[(Row, Long)], Iterator[(Row, Long)]) = iter.duplicate

    // compute the number of instances in each partition
    while (newIterator._2.hasNext) {
      newIterator._2.next()
      length = length + 1
    }

    // the scores
    val scores: Array[(Long, Double)] = new Array[(Long, Double)](length)
    var counter: Int = 0

    // save the value of the iterator
    var aux: (Row, Long) = null

    // look the instances
    while (newIterator._1.hasNext) {
      aux = newIterator._1.next()
      // look the attributes
      scores(counter) = (aux._2, aux._1.toSeq.zipWithIndex.map(att => {
        // index = (attribute - min) / (max - min) / k = ((attribute - min) * k) / (max - min)
        index = (((att._1.asInstanceOf[Double] - limits.value(att._2).head) * k) / (limits.value(att._2).last - limits.value(att._2).head)).asInstanceOf[Int]
        // check if the value is the max, because the index will be equal to histogram size, going out of bounds
        if (index >= k)
          index = k - 1

        // compute the score log(1/h_i) of each attribute
        Math.log(1 / histogram.value(att._2)(index))
      }).sum) // sum the score of each attribute to get the score for each instance
      counter = counter + 1
    }

    Array(scores).iterator
  }

  /**
   * accumulates the scores for the static version
   *
   * @param score1 set of scores 1
   * @param score2 set of scores 2
   * @return the scores accumulated
   */
  private def scoresReduce(score1: Array[(Long, Double)], score2: Array[(Long, Double)]): Array[(Long, Double)] = {
    score1 ++ score2
  }

  /**
   * computes the scores for the dynamic version
   *
   * @param iter      the instances
   * @param histogram the histogram
   * @return the index of each instance and his score
   */
  private def scoresMapDynamic(iter: Iterator[(Row, Long)], histogram: Broadcast[Seq[Array[(Double, Double, Double)]]]): Iterator[Array[(Long, Double)]] = {
    // index to compute the bin of each attribute
    var index: Int = 0

    // number of instances in each partition
    var length: Int = 0

    // auxiliar iterator to compute the number of instances in each partition
    val newIterator: (Iterator[(Row, Long)], Iterator[(Row, Long)]) = iter.duplicate
    // compute the number of instances in each partition
    while (newIterator._2.hasNext) {
      newIterator._2.next()
      length = length + 1
    }

    // the scores
    val scores: Array[(Long, Double)] = new Array[(Long, Double)](length)

    var counter: Int = 0

    // save the value of the iterator
    var aux: (Row, Long) = null

    // control when the bin is found
    var found: Boolean = false

    //look the instances
    while (newIterator._1.hasNext) {
      aux = newIterator._1.next()
      // look the attributes
      scores(counter) = (aux._2, aux._1.toSeq.zipWithIndex.map(att => {
        index = 0
        //find the bin, the value of the attribute must be between max and min value of the bin
        while (!found) {
          if (att._1.asInstanceOf[Double] >= histogram.value(att._2)(index)._1 && att._1.asInstanceOf[Double] <= histogram.value(att._2)(index)._2) {
            found = true
          } else {
            index = index + 1
          }
        }
        found = false

        // compute the score log(1/h_i) of each attribute
        Math.log(1 / histogram.value(att._2)(index)._3)
      }).sum) // sum the score of each attribute to get the score for each instance
      counter = counter + 1
    }

    Array(scores).iterator
  }

  /**
   * Computes the HBOS algorithm
   */

  def fit(): RDD[Double] = {
    if (strategy != "static" && strategy != "dynamic") {
      throw new Exception("Method must be static or dynamic")
    } else {
      val sc = SparkSession.builder().getOrCreate()

      // split the column with the features into a dataset which contains one columns for each attribute
      var data = dataset.select(col = "features")
      val disassembler = new VectorDisassembler()
        .setInputCol("features")
      data = disassembler.transform(data)
      data = data.drop(colName = "features")
      data.persist(StorageLevel.MEMORY_AND_DISK)

      if (strategy == "static") {

        // compute the histograms
        val histograms: Array[Array[Double]] = Array.fill(data.columns.length, n_bins)(0.0)
        val limits: Array[Array[Double]] = Array.fill(data.columns.length, n_bins)(0.0)

        data.columns.zipWithIndex.foreach(name => {
          val histogram = data.select(name._1).rdd.flatMap(v => v.toSeq.map(_.asInstanceOf[Double])).histogram(n_bins)
          limits(name._2) = histogram._1
          histograms(name._2) = histogram._2.map(_ + 1e-12)
          val histogram_sum = histograms(name._2).sum
          histograms(name._2) = histograms(name._2).map(_ / histogram_sum)
        })

        val histogramBroadcast: Broadcast[Array[Array[Double]]] = sc.sparkContext.broadcast(histograms)
        val limitsBroadcast: Broadcast[Array[Array[Double]]] = sc.sparkContext.broadcast(limits)

        // compute the scores
        sc.sparkContext.parallelize(data.rdd.zipWithIndex().mapPartitions(split => scoresMapStatic(split, histogramBroadcast, limitsBroadcast, n_bins)).reduce(scoresReduce)).sortByKey().map(_._2)
      } else {

        // size of the dataset
        val size: Long = data.count()
        // size / k to get the bound of each bin
        val inc: Long = (size / n_bins.asInstanceOf[Double]).ceil.asInstanceOf[Long]
        // control index
        var splitIndex: (Long, Long) = (0, inc)

        // compute the bounds for each bin, giving 0 to values not necessaries, 1 to first value and 2 to last value of the bin
        val index: RDD[Long] = sc.sparkContext.parallelize((0 until size.asInstanceOf[Int]).map(i => {
          if (i >= (size - 1)) { // if size/k is not exact, the last value must be updated
            2.asInstanceOf[Long]
          } else {
            // first value of the bin
            if (i == splitIndex._1) {
              1.asInstanceOf[Long]
              // last value of the bin
            } else if (i == splitIndex._2 - 1) {
              //update control index
              splitIndex = (splitIndex._2, splitIndex._2 + inc)
              2.asInstanceOf[Long]
              // not necessary value
            } else {
              0.asInstanceOf[Long]
            }
          }
        }))

        // join the index computed previously to the dataset, then filter getting the values with index 1 and 2 and save those values
        var sortValues: RDD[(Long, Row)] = data.select(data.columns.head).sort(data.columns.head).rdd.zipWithIndex().map(_.swap)
        data.columns.tail.foreach(name => {
          sortValues = sortValues.join(data.select(name).sort(name).rdd.zipWithIndex().map(_.swap)).map(value => (value._1, Row((value._2._1.toSeq :+ value._2._2.get(0)): _*)))
        })
        val organizeValues: Array[Array[Double]] = sortValues.join(index.zipWithIndex().map(_.swap)).filter(_._2._2 > 0).map(value => value._2._1.toSeq.map(_.asInstanceOf[Double]).toArray).collect().transpose

        val histograms: Array[Array[(Double, Double, Double)]] = Array.fill(data.columns.length, n_bins)(0.0, 0.0, 0.0)
        var counter: Int = 0
        var first = true
        var firstValue: Double = 0

        // compute the histogram, pairing first and last value, also save the value of the features
        organizeValues.zipWithIndex.foreach(values => {
          // compute histogram for each attribute
          counter = 0
          values._1.sorted.foreach((value: Double) => {
            if (first) {
              firstValue = value
              first = !first
            } else {
              histograms(values._2)(counter) = (firstValue, value, inc / (value + 1e-9 - firstValue))
              counter = counter + 1
              first = !first
            }
          })
        })

        // compute the scaled histogram
        histograms.indices.foreach(i => {
          val histogram_sum = histograms(i).map(_._3).sum
          histograms(i) = histograms(i).map(values => (values._1, values._2, values._3 / histogram_sum))
        })

        val histogramBroadcast: Broadcast[Seq[Array[(Double, Double, Double)]]] = sc.sparkContext.broadcast(histograms)

        // computes the scores
        sc.sparkContext.parallelize(data.rdd.zipWithIndex().mapPartitions(split => scoresMapDynamic(split, histogramBroadcast)).reduce(scoresReduce)).sortByKey().map(_._2)
      }
    }
  }
}
