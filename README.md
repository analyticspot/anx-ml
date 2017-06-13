# Motivation

Generally in a machine learning problem one starts with some fairly raw data and then does some pre-processing. That
can involve generating new features, cleaning existing features, doing database lookups, and even making API calls to
external services to augment the data. Thus, feature generation is really a directed graph (a DAG) where you start with
raw data, generate new data from that, etc.

Additionally, some "features" aren't really features, but just intermediate bits of data. These data types often aren't
numerical, categorical, or ordinal and thus don't conform to standard machine learning data types. For example, for text
processing we might parse sentences and generate a "parse tree" indicating parts of speech tags for each word and the
relationships between the words (e.g. "this adjective modifies that verb"). Other nodes in the graph would then extract
features from the parse tree like the number of verbs in the text or the number of unique nouns.

Of course, we want to be able to both train models and deploy them. That means we need a mechanism to serialize the
entire DAG after training and then deserialize it in production. We would like to be able to use the deserialized graph
to make predictions on data from a variety of sources: files in various formats, in-memory data (e.g. data received by a
web server), Spark RDDs, etc.

While there are several machine learning systems for Java,they all lack some properties we want for our system. Our
desired capabilities are:

* We want to compute each piece of information exactly once even if multiple data transforms use the same data as input.
* Some data generators are slow and may be asynchronous (e.g. API calls). We don't want to have to worry about when
  then inputs to a computation are available; the framework should manage that for us.
* We want to be able to serialize the entire trained DAG to a single file and then deserialize it in a different
  executable and apply it.
* We want to be able to inject alternative implementations of some transforms when we deserialize. For example, we might
  obtain some data from a database during training but need to obtain that same data via an API call in production so we
  need to be able to swap out the implementation when we deserialize our trained DAG.
* We want everything to be strongly typed: `DataTransform` instances should be able to declare the types they can
  consume and produce and `DataSet` instances should have method to retrieve some of their columns in a type safe way.
* We want to allow `DataTransform` types that can produce multiple outputs even if the number of outputs can't be known
  until the transform has been trained (e.g. a bag of words transform can't know how many words are in the vocabulary
  until it has seen the training data).
* We want to be able to run independent parts of the DAG in parallel utilizing all the CPUs on the machine.

The DataGraph framework described here is intended provides all of these features.

# Quick Example

Consider the following training task: Our input is unstructured text: a movie review. Our target is a value in the
range 0 - 4 representing the number of stars the reviewer gave. We want to learn to predict the number of stars from
the full text of the review. In order to do this we want to generate several features and transform the inputs.

Specifically, we want to:

1. Convert the input review into a "bag of words" representation: one row for each review, one column for each
   unique word, and the value at the row/column intersection is the number of times that word appeared in that review.
   For memory efficiency we want to use a `SparseMatrix` class rather than a `double[][]` since most values in the
   matrix will be 0.
2. We want to convert the bag of words into a lower-dimensional vector via LSA.
3. For each of the 4 possible ratings we want to compute a centroid for the vectors computed in (2) indicating a
   "typical" review for that rating.
4. We then want to generate the following features
    1. For each review we want to compute its distance from each of those centroids thus generating 5 features.
    2. We have a list of "positive words" that contains things like "good", "excellent", etc. For each review we want to
       count the number of times words in this list have appeared.
    3. Similarly, we'd like to compute the number of "negative words" in each review.
5. We then want to send the 7 features above to a linear regression algorithm

This task would look something like the following in Java given this framework:

```java
// The DataGraph represents the complete pipeline described above.
DataGraph.GraphBuilder builder = DataGraph.builder();

// ColumnId instances let you obtain data in a type-safe way. 
ColumnId<String> reviewColId = new ColmnId<>("review", String.class);
ColumnId<Integer> numStartsColId = new ColumnId<>("numStart", Integer.class);

// Describes the source. Unlike other nodes the source is just a description so we can later pass any values that
// conform to the description. numStarsId is a "train-only" value because it is not required to make predictions.
GraphNode source = builder.source()
    .addColumn(reviewColId)
    .addTrainOnlyColumn(numStarsColId)
    .build()

// Now we consume the output of the source node and transform it into a Bag of words representation. Since we don't
// know how many words there will be until we train we use a ColumnIdGroup.
ColumnIdGroup<Integer> bagOfWordsGroupId = new ColumnIdGroup<>("words", Integer.class);
GraphNode bagOfWords = builder.addTransform(source, new BagOfWords(bagOfWordsGroupId));


// Now we do some dimension reduction on the bag of words representation
ColumnIdGroup<Double> lsaGroupId = new ColumnIdGroup<>("lsa", Double.class);
GraphNode lsa = builder.addTransform(bagOfWords, new LsaTransform(lsaGroupId));

// Now compute centroids (during training) and compute the distance between the lsa vector for each review. This
// will have 5 outputs: one for each star rating. Note that it depends on both the output of lsa and the source,
// though the source is only required during training.
GraphNode distFromCentroid = builder.addTransform(lsa, source, new DistFromCentroidTransform(numStarsColId));


// Add our positive and negative word counts.
GraphNode posWords = builder.addTransform(bagOfWords, new WordCount(positiveWordList)));

GraphNode negWords = builder.addTransform(bagOfWords, new WordCount(negativeWordList));

// Merge together the 3 data sets we'll use for our predictions
GraphNode merged = builder.merge(distFromCentroid, posWords, negWords);

// Add our classifier
ColumnId<Integer> predictionId = new ColumnId<>("prediction", Integer.class);
GraphNode regression = builder.addTransform(merged, new LinearRegression(predictionId));

// Tell the graph builder that the output of the classifier is the result of the entire graph.
builder.setResult(regression);

// And build the graph.
DataGraph graph = builder.build();
```

Note that DataGraph is written in Kotlin so while it is 100% compatible with Java, Kotlin users can use Kotlin's builder
DSL which is much less verbose than the above.

Given the above we can train the entire DAG. This means that the bag of words component will learn the allowed
vocabulary, the LSA will learn the appropriate dimension reduction, the centroid computation will learn the centroids
for each rating, etc. To train the pipeline just call `graph.trainTransform()`.

Now that we've got our trained model we'd like to deploy it to production. This means we need to serialize the entire
graph representation and what each of the learning nodes learned:

```java
GraphSerDeser serDeser = new GraphSerDeser();
serDeser.serialize(graph, "/path/to/file");
```

Finally, we're ready to deploy to production. 

```java
DataGraph dg = serDeser.deserialize("path/to/trained");

// Now we can make predictions.
Observation toPredict = dg.createSource("the text of a movie review");
CompletableFuture<DataSet> resultFuture = dg.transform(toPredict, Executors.newFixedThreadPool(4));
```

Note that calls to the `DataGraph` are asynchronous. This is because each node in the DAG can, if it wants, compute its
result asynchronously. This allows nodes that do things like call external APIs to augment the data to do so without
blocking a thread.

`DataGraph` implements the `LearningTransform` interface and so can be used just like a single transform in another
`DataGraph`. This allows you to decompose a large graph into several smaller ones. It also allows you to use the
deserialization injection on entire sub-graphs. See the `SERIALIZATION.README.md` file for details.

# Overview

The library consists of several components:

* `DataSet`: An immutable collection of data passed from transform to transform. Each transform takes one or more
  `DataSet` instances as input and produces exactly one `DataSet` instance as output. The `DataSet` contains one
  row per data instance and can contain an arbitrary number of columns of any data type.
* `ColumnId`: The id used to retrieve a single data item or entire column of data from a `DataSet`. A column id consists
  of a name, which is just a `String`, and a type which keeps everything type safe.
* `ColumnIdGroup`: represents several `ColumnId` instances whose names all have a common prefix and all of which have
  the same type. These are generally used with transformations that learn how many columns they will create during
  training. For such transforms we can't know the id's of each column (since we can't even know how many columns there
  will be), but we can have a single `ColumnIdGroup` that lets us refer to all the columns it will produce.
* `DataTransform`: These can be straight transformations (e.g. converting Strings to lowercase), or supervised or
  unsupervised learners.
* `GraphNode`: a `DataTransform` plus some other information like which other transforms feed data to this transform,
  which transforms consume the output of this transform, etc.
* `DataGraph`: a DAG of `GraphNode` instances.
* `GraphSerDeser`: used for serializing and deserializing a graph. This allows for a variety of serialization formats so
  that we can interoperate with other machine learning libraries and allows for injection. See the
  [SERIALIZATION.README](./SERIALIZATION.README.md) file for details.

## DataSet

A [`DataSet`](./framework/src/main/kotlin/com/analyticspot/ml/framework/dataset/DataSet.kt) is simply a collection of
data. Individual data items can be access via calls like `dataSet.value(rowIndex, columnId)` where `rowIndex` is a
0-based row index and `columnId` is a `ColumnId` instance. The `value` method is generic and will return a whose type
matches the type of the `ColumnId` instance. Thus, for example, if `columnId` is a `ColumnId<ComplexDataType>` then
`dataSet.value(0, columnId)` will return the an instance of `ComplexDataType` from the first row of data.

Note that `DataSet` is "column oriented" a immutable so that operations which select a subset of columns or merge the
columns from multiple data sets are extremely efficient. Similarly, it is easy and efficient to obtain an entire column
of data. This allows us to write very concise and efficient data transforms that operate on a few columns. For example,
the following Kotlin code is a valid `DataTransform` that extracts a single `String` column (`columnToConvert`) from a
`DataSet` and produces a new `DataSet` with one column (`newColumn`) that contains the same contents but converted to
lower case:

```kotlin
class ConvertToLower(
    val columnToConvert: ColumnId<String>, val newColumn: ColumnId<String>) : SingleDataTransform {
    
    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val result = DataSet.build {
            addColumn(newColumn, dataSet.column(columnToConvert).mapToColumn { it.toLowerCase() })
        }
        return CompletableFuture.completedFuture(result)
    }

}
```

# Bridges

A machine learning framework is much more useful if there is a library of existing machine learning algorithms that can
be used with it. To enable this we have "bridges" that allow one to easily use code from libraries like
[Smile](https://haifengl.github.io/smile/) and [DeepLearning4j](https://deeplearning4j.org/) with DataGraph. For
example, converting a Smile "soft classifier" (a classifier that produces posterior probabilities) into a DataGraph
`DataTransform` can be accomplished with just a few lines of Kotlin code like:

```kotlin
// Takes an array of Smile Attribute instance representing the columns from the input DataSet
// and returns a Smile SoftClassifier instance -- in this case a decision tree.
val dtTrainerFactory = { attrs: Array<Attribute> -> DecisionTree.Trainer(attrs, maxNodes) }
// Use the factory above to create a DataGraph compatible DataTransform
val treeTransform = SmileSoftClassifier(trainData.targetId, dtTrainerFactory)
```

We have been adding bridges for frameworks and parts of frameworks as we've needed them so the list is still fairly
small. However, creating new ones is fairly easy. All bridges can be found in the `bridges` subdirectory.

Note that the bridges are published as separate Maven artifacts so that users who don't want to use a framework need not
have any dependencies on that framework. The existing Maven artifacts can be found on
[jcenter](https://bintray.com/oliverdain/ANX/ANX-ML-Framework).

# ExecutorService

The `DataGraph` methods like `transform`, `trainTransform` that execute the DAG an argument of type `ExecutorService`.
All `DataTransform` methods that are called by the graph execution and guaranteed to be run on a thread managed by this
`ExecutorService`. In addition, the `ExecutorService` is passed to the `transform` and `trainTransform` methods of each
`DataTransform` in the DAG. Thus, if the transform is going to some computationally expensive work in a single thread
they can simply run their computation directly in the method and ignore the `ExecutorService`. However, computations
that can be parallelized are free to submit work to the `ExecutorService`. Since this is the same `ExecutorService` that
manages the execution of the `DataGraph` the submitted work will be interleaved with the work of other `DataTransform`
instance in the same `DataGraph`. This allows us to do things like create one thread per CPU and use that thread pool
for all computationally expensive work parallelizing not only the execution of individual transforms but also the work
done by all transforms that can execute concurrently (those transforms that have had their inputs computed).

# Output Interceptors

[OutputInterceptor](./framework/src/main/kotlin/com/analyticspot/ml/framework/datagraph/OutputInterceptor.kt) instances
can be passed to the `transform` and `trainTransform` methods in order to inspect, and optionally modify, the output of
any `GraphNode` before that output is seen by any other node in the graph. This can be handy for debugging or performing
experiments (e.g. "how much would it affect things if the value of feature X was always 11?").

# Using

To use, you can download the source here and build it by running `./gradlew check`. Alternatively, you can download
pre-built jar files from jcenter. The maven coordinates for the main framework are:

* groupId: com.analyticspot.ml
* artifactId: framework
* version: the `VERSION` file in this directory contains the latest version number.

For gradle, this means a dependency like `compile 'com.analyticspot.ml:framework:0.1.1'` will pull the correct jar.
