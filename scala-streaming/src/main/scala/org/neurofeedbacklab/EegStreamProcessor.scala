package org.neurofeedbacklab

import edu.ucsd.sccn.LSL
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object EegStreamProcessor {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("EEG LSL Structured Streaming")
      .getOrCreate()
    import spark.implicits._

    // 1. Resolve LSL inlet
    val inlet = {
      val info = LSL.resolve_stream("name", "EEGWin", 1, 5)[0]
      new LSL.StreamInlet(info)
    }

    // 2. Simple infinite collection into Spark micro-batches
    //    (pull a chunk per trigger interval)
    val schema = "timestamp DOUBLE, channel INT, value FLOAT"
    val source = spark.readStream
      .format("rate")             // Generates triggers; we override with custom data pull
      .option("rowsPerSecond", 1)
      .load()
      .mapPartitions{ _ =>
        val chunk = Array.ofDim[Float](inlet.info().channel_count())
        Iterator.continually{
          val ts = inlet.pull_chunk(chunk)
          chunk.zipWithIndex.map{ case(v,i)=>(ts,i,v) }
        }.flatten
      }(org.apache.spark.sql.Encoders.tuple(Encoders.scalaDouble,Encoders.scalaInt,Encoders.scalaFloat))
      .toDF("timestamp","channel","value")

    // 3. Placeholder processing – compute rolling mean as artifact suppressor
    import org.apache.spark.sql.expressions.Window
    import org.apache.spark.sql.DataFrame

    // Use Spark SQL window function for rolling mean (if needed)
    // Here, we just use groupBy with time window for demonstration
    val processed = source
      .groupBy(window($"timestamp", "10 seconds"), $"channel")
      .agg(avg($"value").as("avg_val"))

    // 4. Sink to console / EBS writer
    val query = processed.writeStream
      .outputMode("update")
      .format("console")
      .start()

    query.awaitTermination()
  }
}
