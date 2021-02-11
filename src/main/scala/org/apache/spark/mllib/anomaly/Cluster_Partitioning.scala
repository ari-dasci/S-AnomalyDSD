package org.apache.spark.mllib.anomaly

import org.apache.spark.ml.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

/**
 * Cluster-based partitioning
 *
 * @param data        the dataset in Dataset[Row] format with the features grouped in "features" column
 * @param clus_method clustering method for the local neigborhood calculation: "kmeans", "bisec" (default = "kmeans")
 * @param n_clus      number of clusters for the local neigborhood calculation (default = 11)
 * @param max_size    Maximum number of elements per partition
 * @param min_size    Minimum number of elements per partition
 * @return a Dataset[Row] repartitioned according to the previous conditions
 */

class Cluster_Partitioning(val data: Dataset[Row], val clus_method: String = "kmeans", val n_clus: Int = 10, val max_size: Int = 10000, val min_size: Int = 1000) extends Serializable {

  // Clustering method initialization
  val method = clus_method match {
    case "kmeans" => new KMeans().setK(n_clus)
    case "bisec" => new BisectingKMeans().setK(n_clus)
  }

  // Clustering fit
  val model = method.fit(data.select(col = "features"))
  val dataClustered: DataFrame = model.transform(data)

  // Clustering repartition
  val predictions: DataFrame = dataClustered.repartitionByRange(dataClustered("prediction"))
  predictions.persist(StorageLevel.MEMORY_AND_DISK)
  val columns: Array[String] = predictions.columns
  val max_clus: Int = predictions.agg(aggExpr = "prediction" -> "max").collect()(0).getInt(0) + 1

  /**
   * Data balancing according to the cluster asigned to each instance:
   * If the partition has more than max_size, it is splitted in partitions of size max_size
   * Partitions with less than min_size elements are joined together
   */

  private def checkType[T](v: T) = v match {
    case _: org.apache.spark.ml.linalg.DenseVector => "Dense"
    case _: org.apache.spark.ml.linalg.SparseVector => "Sparse"
    case _ => "Unknown"
  }

  def balance_clusters(): Dataset[Row] = {
    val data_balanced = predictions.rdd.mapPartitions { partition =>
      val temp = partition.toArray
      val cluster = temp.map { l =>
        val features_temp = l.get(l.fieldIndex(name = "features"))
        if(checkType(features_temp) == "Dense") features_temp.asInstanceOf[org.apache.spark.ml.linalg.DenseVector].toArray :+ l.getAs[Long](fieldName = "index").toDouble :+ l.getAs[Int](fieldName = "prediction").toDouble
        else features_temp.asInstanceOf[org.apache.spark.ml.linalg.SparseVector].toArray :+ l.getAs[Long](fieldName = "index").toDouble :+ l.getAs[Int](fieldName = "prediction").toDouble
      }
      val scores = temp.map { l => l.getAs[org.apache.spark.ml.linalg.DenseVector](fieldName = "scores").toArray }
      if (cluster.length > 0) {
        val cluster_size = cluster.length
        val id_clus = cluster(0).last
        var split_count = 1
        if (cluster_size > max_size) {
          for (i <- 0 until cluster_size by max_size) {
            if (i + max_size > cluster_size) {
              val new_clust = id_clus + (split_count / scala.math.pow(10, split_count.toString.length))
              (i until cluster_size).foreach { l =>
                cluster(l)(cluster(l).length - 1) = new_clust
              }
            } else {
              val new_clust = id_clus + (split_count / scala.math.pow(10, split_count.toString.length))
              (i until i + max_size).foreach { l =>
                cluster(l)(cluster(l).length - 1) = new_clust
              }
              split_count = split_count + 1
            }
          }
        } else {
          if (cluster_size < min_size) {
            if (min_size != -1) {
              cluster.map { l => l.init :+ max_clus }
            }
          } else {
            cluster.map { l => l.init :+ id_clus }
          }
        }
        cluster.zipWithIndex.map { case (l, k) =>
          (l.apply(l.length - 2).toLong, Vectors.dense(l.init.init), Vectors.dense(scores(k)), l.last.toInt)
        }.toIterator
      } else {
        cluster.zipWithIndex.map { case (l, k) =>
          (l.apply(l.length - 2).toLong, Vectors.dense(l.init.init), Vectors.dense(scores(k)), l.last.toInt)
        }.toIterator
      }
    }

    import data.sqlContext.implicits._
    data_balanced.toDF(colNames = columns: _*).repartitionByRange(partitionExprs = $"prediction")
  }
}
