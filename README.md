# Motivation

Generally in a machine learning problem one starts with some fairly raw data and then does some pre-processing. That
can involve generating new features, cleaning existing features, doing database lookups, and even making API calls to
external services to augment the data. Thus, feature generation is really a directed graph (a DAG) where you start with
raw data, generate new data from that, then yet more new data that may depend on data generated in an earlier step, etc.

Additionally, some "features" aren't really features, but just intermediate bits of data. For example, for text
processing we might generate a matrix for the standard "bag of words" representation but we probably wouldn't use this
directly. Instead we might do dimension reduction on that data. We thus talk about *data generators*, which can create
arbitrary data types. Each data generator can restrict the input types it understands.

This presents several challenges:
* We want to compute each piece of information exactly once even if multiple data generators use the same data as input.
* If some data isn't required to make a prediction we may not want to generate it. For example, if we were using a
decision tree as our classifier, a given item might take a path through the tree such that certain features are never
used. This is particularly useful for data generators which make API calls that may cost money.
* Some data generators are slow and may be asynchronous. We don't want to have to worry about which data is available
when and thus know when it's safe to run the next feature/data generator in the chain.

The feature framework here is intended to solve these issues.

# Quick Example

Consider the following training task: Our input is unstructured test: a movie review. Our target is a value in the
range 0 - 4 representing the number of stars the reviewer gave. We want to learn to predict the number of stars from
the full text of the review. In order to do this we want to generate several features and transform the inputs.
Specifically, we want to:

1. Convert the input review into a "bag of words" representation: one row for each review, one column for each
unique word, and the value at the row/column intersection is the number of times that word appeared in that review.
2. We want to convert the bag of words into a lower-dimensional vector via LSA (basically Pincipal Components Analysis).
3. For each of the 4 possible ratings we want to compute a centroid for the vectors computed in (2) indicating a
"typical" review for that rating.
5. We then want to generate the following features
    1. For each review we want to compute its distance from each of those centroids thus generating 5 features.
    2. We have a list of "positive words" that contains things like "good", "excellent", etc. For each review we want to
compute the number of times words in this list has appeared.
    3. Similarly, we'd like to compute the number of "negative words" in each review.
7. We then want to send the 7 features above to a linear regression algorithm

This task would look something like the following in Java given this framework:

```java
// The DataGraph represents the complete pipeline described above. It is of type Double as it's final output will be
// a dobule - the predicted rating.
DataGraph<Double> dg = new DataGraph();
// This enum allows us to do a kind of injection so that, for example, we can use flat files to train
// but data obtained via REST endpoint for predictions. 
enum Injectables {
  REVIEW_SOURCE
};

// A DataToken refers to a specifc source or data generator so we can declare that other generators depend on it.
DataToken<String> inputSource = dg.addSource(new FlatFileGenerator("path/to/reviews"), Injectables.REVIEW_SOURCE);

// The bag of words generator depends on the input source so it uses that token to declare the dependency. This computes
// an array of doubles for each input so it's type is Array<Double>.
DataToken<Array<Double>> bagOfWords = dg.addGenerator(new BagOfWordsGenerator(), inputSource);

DataToken<Array<Double>> lsa = dg.addGenerator(new Lsa(), bagOfWords);

DataToken<Array<Double>> reviewCentroids = db.addGenerator(new CentroidComputer(), lsa);

DataToken<Array<Double>> distanceFromCentroid = db.addGenerator(new DistFromVector(), reviewCentroids);

DataToken<Integer> numPositiveWords = db.addGenerator(new WordCount(positiveWordList), bagOfWords);

DataToken<Integer> numNegativeWords = db.addGenerator(new WordCount(negativeWordList), bagOfWords);

// Finally we add our classifier. It wants a vector of doubles for each row so we have to take all the algorithm
// inputs and change their representation.
DataToken<Array<Double>> vectorOfFeatures = db.addGenerator(new DoubleRepr(),
    distanceFromCentroid, numPositiveWords, numNegativeWords);

// And finally we add the classifier.
db.addPredictor(new LinearRegression(), vectorOfFeatures);
```

Note that, thanks to Kotlin's builder DSL the above looks a bit nicer in Kotlin.

Given the above we can train the entire pipeline. This means that the bag of words component will learn the allowed
vocabularly, the LSA will learn the appropriate dimension reduction, the centroid computation will learn the centroids
for each rating, etc. To train the pipeline just call `dg.train()`.

Now that we've got our trained model we'd like to deploy it to production. This means we need to serialize all the
model paramters that were learned: `dg.serialize("/path/to/trained")`. Finally, we're ready to deploy to production.
However, we can't use exactly the same classes in production as we did to train. Specifically, the data shouldn't come
from flat files, it should come from a String in memory where the string is sent to us via REST API. Thus we need to
"rebind" the `REVIEW_SOURCE`. Note that we could rebind additional components as well. For example, we might have 
cached expensive computations for training but need to do the computation "for real" in production. The enum we used
above allows us to specify factory functions for each enum value. Thus we can do something like this:

```java
DataGraph<Double> dg = DataGraph.fromFile("path/to/trained");

// Now we can make predictions. Here we bind enums that weren't bound in the fromFile call above, including the
// actual input data.
double prediction = dg.preparePredict()
    .withBinding(REVIEW_SOURCE, new SimpleString(dataRecievedFromTheApi))
    .predict();
```

# Overview

The library consists of several components that will be described in the sections below:

* `SourceGenerator`: little more than an `Iterable`. This represents the "raw" training, test, or validation
set. Here we say "raw" as the source may be heavily transformed before being consumed by a ML algorithm.
* `DataGenerator`: this class generates data of arbitrary type. It may require other `DataGenerator` outputs as as
input.
* `TrainableDataGenerator`: these are `DataGenerator` instances that have a `train` method which allows them to learn
from the data. This allows us to treat unsupervised algorithms like clustering or dimension reduction as a
`DataGenerator`. Furthermore, classification and regression algorithms can then also be treated as
`TrainableDataGenerator`. They are simple data generators that take several other generator outputs as input and the
data they produce is a prediction.
* `DataGraph`: this describes the DAG of `DataGenerator` and `SourceGenerator` instances. This is how we define the
inputs to each component and define which `DataGenerator` instances represent the final output.
* `RepresentationTransformer`: a `Collection<Feature<?>>` isn't a very convenient data representation for most machine
learning algorithms. However, there is no single "best" representation. For example, a matrix of doubles is probably 
best for a linear regression algorithm but column vectors of fixed type are probably preferable for decision trees and
rule learners (here each column would be the value of a single feature for all items in the training set). Thus we
employ representation transformers to convert from collections of features to something more convenient for the
training algorithm. These representations are cached so that if several `TrainableDataGenerator` instances want the 
same representation it is computed only once.

## Source Generator

This is just a source of data. It may read from a database, information passed to a service via API, or a CSV file. The
most important thing about a source generator is that it does not have any other `SourceGenerator` or `DataGenerator`
dependencies. This almost always represents the raw input from which we want to learn or make predictions.

# Model Pipeline Serialization

# Injection

We might train with one source and then use a different source.

Serialize a whole `DataGraph`.

Use factory functions so injection compatible.


