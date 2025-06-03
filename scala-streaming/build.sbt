name := "EegStreamProcessor"
version := "0.1.0"
scalaVersion := "2.13.12"
libraryDependencies ++= Seq(
  "org.apache.spark"  %% "spark-sql"         % "3.5.0" % "provided",
  "org.apache.spark"  %% "spark-mllib"       % "3.5.0",
  "edu.ucsd.sccn"     %  "labstreaminglayer" % "1.16"  // liblsl-java
)
