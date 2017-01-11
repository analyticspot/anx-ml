# Motivation

Generally in a machine learning problem one starts with some fairly raw data and then does some pre-processing. That
can involve generating new features, cleaning existing features, doing database lookups, and even making API calls to
external services to augment the data. Thus, feature generation is really a directed graph (a DAG) where you start with
raw data, generate new data from that, then yet more new data that may depend on data generated in an earlier step, etc.

Additionally, some "features" aren't really features, but just intermediate bits of data. For example, for text
processing we might generate a matrix for the standard "bag of words" representation but we probably wouldn't use this
directly. Instead we might do dimension reduction on that data. 

Each node is the DAG is a `DataTransform`. A `DataTransform` must have a `transform` method which accepts a `DataSet`
and produces a new `DataSet` as output. Additionally, some `DataTrasform` instances can learn from training data. These
transforms have a `trainTransform` method that first learns from the data and then transforms it according to what was
learned. Once `trainTransform` has been called `transform` can be called on new data to apply what was learned from
the training data.

There are several machine learning systems for Java, but they all lack some properties we want for our system. Our
desired capabilities are:

* We want to compute each piece of information exactly once even if multiple data transforms use the same data as input.
* Some data generators are slow and may be asynchronous. We don't want to have to worry about which data is available
when and thus know when it's safe to run the next feature/data generator in the chain.
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
* If some data isn't required to make a prediction we may not want to generate it. For example, if we were using a
decision tree as our classifier, a given item might take a path through the tree such that certain features are never
used. This is particularly useful for data generators which make API calls that may cost money.

The feature framework here is intended to solve these issues.

# Quick Example

Consider the following training task: Our input is unstructured test: a movie review. Our target is a value in the
range 0 - 4 representing the number of stars the reviewer gave. We want to learn to predict the number of stars from
the full text of the review. In order to do this we want to generate several features and transform the inputs.
Specifically, we want to:

1. Convert the input review into a "bag of words" representation: one row for each review, one column for each
unique word, and the value at the row/column intersection is the number of times that word appeared in that review.
2. We want to convert the bag of words into a lower-dimensional vector via LSA. 
3. For each of the 4 possible ratings we want to compute a centroid for the vectors computed in (2) indicating a
"typical" review for that rating.
5. We then want to generate the following features
    1. For each review we want to compute its distance from each of those centroids thus generating 5 features.
    2. We have a list of "positive words" that contains things like "good", "excellent", etc. For each review we want to
compute the number of times words in this list have appeared.
    3. Similarly, we'd like to compute the number of "negative words" in each review.
7. We then want to send the 7 features above to a linear regression algorithm

This task would look something like the following in Java given this framework:

```java
// The DataGraph represents the complete pipeline described above.
DataGraph.GraphBuilder builder = DataGraph.builder();

// ValueIds let you obtain data in a type-safe way. Each ValueId can be converted into a ValueToken by the node that
// produces the data. A ValueToken is just a ValueId plus some hidden data that allows fast retrieval from the
// DataSet.
ValueId<String> reviewValueId = new ValueId<>("review", String.class);
ValueId<Integer> numStarsId = new ValueId<>()
// Describes the source. Unlike other nodes the source is just a description so we can later pass any values that
// conform to the description. numStarsId is a "train-only" value because it is not required to make predictions.
GraphNode source = builder.source().withValue(reviewValueId).withTrainOnlyValue(numStarsId).build()

// Now we consume the output of the source node and transform it into a Bag of words representation. Since we don't
// know how many words there will be until we train we use a ValueIdGroup.
ValueIdGroup<Integer> bagOfWordsGroupId = new ValueIdGroup<>("words", Integer.class);
GraphNode bagOfWords = builder.addTransform(source, new BagOfWords(source.token(reviewValueId), bagOfWordsGroupId));


// Now we do some dimension reduction on the bag of words representation
ValueIdGroup<Double> lsaGroupId = new ValueIdGroup<>("lsa", Double.class);
GraphNode lsa = builder.addTransform(bagOfWords,
    new LsaTransform(bagOfWords.tokenGroup(bagOfWordsGroupId), lsaGroupId));

// Now compute centroids (during training) and compute the distance between the lsa vector for each review. This
// will have 5 outputs: one for each star rating. Note that it depends on both the output of lsa and the source.
GraphNode distFromCentroid = builder.addTransform(lsa, source,
    new DistFromCentroidTransform(lsa.tokenGroup(lsaGroupId), stars));


// Add our positive and negative word counts.
GraphNode posWords = builder.addTransform(bagOfWords,
    new WordCount(positiveWordList, bagOfWords.tokenGroup(bagOfWordsGroupId)));

GraphNode negWords = builder.addTransform(bagOfWords,
    new WordCount(negativeWordList, bagOfWords.tokenGroup(bagOfWordsGroupId)));

// Merge together the 3 data sets we'll use for our predictions
GraphNode merged = builder.merge(distFromCentroid, posWords, negWords);

// Add our classifier
ValueId<Integer> predictionId = new ValueId<>("prediction", Integer.class);
GraphNode regression = builder.addTransform(merged, new LinearRegression(predictionId));

// Tell the graph builder that the output of the classifier is the result of the entire graph.
builder.setResult(regression);

// And build the graph.
DataGraph graph = builder.build();
```

Note that, thanks to Kotlin's builder DSL the above looks a bit nicer in Kotlin.

Given the above we can train the entire pipeline. This means that the bag of words component will learn the allowed
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
Observation toPredict = dg.buildObservationFromSource("the text of a movie review");
DataSet result = dg.transform(toPredict, Executors.newFixedThreadPool(4));
```

# Overview

The library consists of several components:

* `ValueId`: The id used to retrieve a single data item. A value id consists of a name, which is just a `String`, and
a type which keeps everything type safe.
* `ValueToken`: A `ValueId` plus some hidden information that allows for fast retrieval of data. This is so that we can
have several `DataSet` subclasses to allow for things like zero-copy merging, etc.
* `Observation`: A single "row" of data. An `Observation` consists of multiple values which can be retrieved via a
`ValueToken`.
* `DataTransform`: These can be straight transformations (e.g. converting Strings to lowercase), or supervised or 
unsupervised learners.
* `GraphNode`: a `DataTransform` plus some other information like which other transforms feed data to this transform,
which transforms consume the output of this transform, etc.
* `DataGraph`: a DAG of `GraphNode` instances.
* `GraphSerDeser`: used for serializing and deserializing a graph. This allows for a variety of serialization formats
so that we can interoperate with other machine learning libraries and allows for injection. See the
`SERIALIZATION.README.md` file for details.
 
