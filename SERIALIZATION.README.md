# Overview

After constructing and training a `DataGraph` we'd like to be able to serialize it. The most common use case for this
is probably to serialize a trained model for deployment. Our requirements for this are:

* Simple: it should be easy to deserialize an entire `DataGraph` and start using it to make predictions.
* Format independent: while most of our own code serializes to JSON we would like to be able to use existing code that
  might serialize in a different formatClass. For example, the Smile project contains a large number of algorithms that
  we might want to use but they serialize in a binary format (Java serialization format).
* Injectable: we'd like to be able to use one implementation for a node when training and a different implementation
  when running in production. For example, we might make an API call to obtain a zip code for an address in production,
  but when training we'd use values stored in a database. In addition, this injection needs to be flexible:
  * Synchronous vs. asynchronous: a graph node might produce its data synchronously when training and asynchronously
    in production. Returning to our zip code resolution example, the data is immediately available when training but
    becomes asynchronous when we need to make an API call to obtain it in production.
  * Not class based: It is quite common to use the same algorithm multiple times in the same graph, but we might not
    want to inject all instances in the graph in the same way. Thus we need some way to tag the individual nodes so we
    can specify the implementation for individual node **instances** at deserialization time.

# Format Summary and Example

A `DataGraph` is serialized as a zip file. There is a file in the zip called `graph.json` that defines the graph. Each
node in the graph has an id. However, the serialization of the node itself is in another file in the zip whose prefix is
the id of the node whose data it contains. That allows us to easily mix binary and text serialization formats.

For example, the `graph.json` file might look like this:

```json
{
  "sourceId": 0,
  "resultId": 2,
  "graph": [
    {
      "class": "com.analyticspot.ml.framework.serialization.GraphSerDeser$SourceSerGraphNode",
      "id": 0,
      "subscribers": [1],
      "columns": [
        {
          "name": "src",
          "clazz": "java.lang.Integer"
        }
      ]
    },
    {
      "class": "com.analyticspot.ml.framework.serialization.GraphSerDeser$TransformSerGraphNode",
      "id": 1,
      "sources": [0],
      "subscribers": [2],
      "metaData": {
        "class": "com.analyticspot.ml.framework.serialization.StandardJsonFormat$MetaData",
        "transformClass": "com.analyticspot.ml.framework.datagraph.AddConstantTransform",
        "formatClass": "com.analyticspot.ml.framework.serialization.StandardJsonFormat"
      }
    },
    {
      "class": "com.analyticspot.ml.framework.serialization.GraphSerDeser$TransformSerGraphNode",
      "id": 2,
      "sources": [1],
      "metaData": {
        "class": "com.analyticspot.ml.framework.serialization.WekaFormatMetaData",
        "transformClass": "com.analyticspot.ml.wrappers.WekaWrapper",
        "formatClass": "com.analyticspot.ml.framework.serialization.WekaFormat"
      }
    }
  }
}
```

Node 2 is a (theoretical) wrapper class that wraps any Weka classifier as a `DataTransform`. To see the actual data
to deserialize we'd have to look at the file named "2" in the zip file and the `Weka` `FormatModule` (see below) would
know how to deserialize that properly.

Note that in the JSON, `graph` is a list in toplological order. Thus, for any node `X`, all of `X`'s sources appear in
the file before `X` does.

# Deserialization

In order to support interoperability with other machine learning libraries we do not specify a single serialization
formatClass for the `DataTransform`s. For example, the Weka project has a huge number of classifiers and filters that we
might want to use but they all serialize to a binary formatClass. This is why the graph structure is serialized as JSON
but the details for each node are in their own files in the same `.zip` file. This way each node can serialize itself it
the way that it sees fit.

However, to deserialize we need to know what is capable of deserializing each node. Thus, each node in the `graph.json`
specifies some `metaData` which corresponds to a `Format`. The `metaData` block is deserialized using polymorphic
deserialization so that the correct subclass of `FormatMetaData` is returned. The `FormatMetaData` then tells the
`GraphSerDeser` what `Format` to use to deserialize the data.

# Injection

`GraphSerDeser` has a `registerFactoryForLabel` method that lets you register a custom `TransformFactory` to be used
for nodes that were serialized with the label of your choice (labels are specified when you build the `DataGraph`).
This allows you to deserialize any `DataTransform` into any arbitrary class that implements the `DataTransform` API.
Since all `DataTransform` instances have an asynchronous API, even if they're synchronous, you can easily convert a
synchronous `DataTransform` into an asynchronous one and vice versa. Furthermore, your `TransformFactory` can be
constructed using the injection library of your choice so that things like database connections and API srcTokens can
be injected into your deserialized `DataTransform` instances.

## Injection Notes
 
Note that this framework does not enforce any class hierarchy. A `TransformFactory` can return any `DataTransform` that
is compatible with the graph even if the implementation returned does not share any interface or base class with the
serialized version (other than both being `DataTransform`).
