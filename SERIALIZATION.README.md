# Overview

After constructing and training a `DataGraph` we'd like to be able to serialize it. The most common use case for this
is probably to serialized a trained model for deployment. Our requirements for this are:

* Simple: it should be easy to deserialize an entire `DataGraph` and start using it to make predictions.
* Format independent: while most of our own code serialized to JSON we would like to be able to use existing code that
  might serialize in a different format. For example, the Weka project contains a huge number of algorithms that we
  might want to use but they serialize in a binary format (Java serialization format).
* Injectable: we'd like to be able to use one implementation for a node when training an a different implementation when
  running in production. For example, we might make an API call to obtain a zip code for an address in production, but
  when training we'd use values stored in database. In addition, this injection needs to be flexible:
  * Synchronous vs. asynchronous: a graph node might produce it's data synchronously when training and asynchronously
    in production. Returning to our zip code resolution example, the data is immediately available when training but
    becomes asynchronous when we need to make an API call to obtain it in production.
  * Immediate vs. on demand: as noted in the main `README` we can have values in an `Observation` which are only
    computed when necessary. Such values are called *on demand values*. During training all values are immediately
    available but may become *on demand* in production.
  * Not class based: It is quite common to use the same algorithm multiple times in the same graph, but we might not
    want to inject all instances in the graph in the same way. Thus we need some way to tag the individual nodes so we
    can specify the implementation for individual node **instances** at deserialization time. While not a requirement,
    it would be nice if this tagging could be re-used to obtain information like training results from these nodes.
    
# Format Summary and Example

A `DataGraph` is serialized as a zip file. There is a file in the zip called `graph.json` that defines the graph. Each
node in the graph has an id. However, the serialization of the node itself is in another file whose name is the id of
the node whose data it contains. That allows us to easily mix binary and text serialization formats.

For example, the `graph.json` file might look like this:

```json
{
    "0": {
        "subscribers": [1],
        "type": "com.analyticspot.ml.framework.datagraph.SourceGraphNode",
        "tokens": [
             {
                 "name": "sourceText",
                 "type": "java.lang.String"
             }
        ]
    },
    "1": {
        "tag": "foo",
        "sources": [0],
        "subscribers": [2],
        "type": "com.analyticspot.ml.framework.datagraph.TransformGraphNode",
        "format": {
            "type": "com.analyticspot.ml.serialization.StandardJson",
            "class": "com.analyticspot.ml.transform.PositiveWordCount"
        },
        "tokens": [
            {
                "name": "positiveWordCount",
                "type": "java.lang.Integer"
            }
        ]
    },
    "2": {
        "sources": [1],
        "type": "com.analyticspot.ml.framework.datagraph.TransformGraphNode",
        "format": {
            "type": "com.analyticspot.ml.serialization.Weka"
        },
        "tokens": [
            {
                "name": "prediction",
                "type": "java.lang.Double"
            }
        ]
        
    },
    "source": 0,
    "result": 2
}
```

Node 2 is a (theoretical) wrapper class that wraps any Weka classifier as a `DataTransform`. To see the actual data
to deserialize we'd have to look at the file named "2" in the zip file and the `Weka` `FormatModule` (see below) would
know how to deserialize that properly.

# Deserialization

In order to support interoperability with other machine learning libraries we do not specify a single serialization
format for the `DataTransform`s. For example, the Weka project has a huge number of classifiers and filters that we
might want to use but they all serialize to a binary format. This is why the graph structure is serialized as JSON but
the details for each node are in their own files in the same `.zip` file. This way each node can serialize itself it
the way that it sees fit.

However, to deserialize we need to know what is capable of deserializing each node. Thus, each node in the `graph.json`
specifies a `format` which corresponds to a `FormatModule`. The `format` block is deserialized using polymorphic
deserialization so that the correct subclass of `FormatModuleData` is returned.

The API for a `FormatModule` is quite simple:

```kotlin
interface FormatModule<T : FormatModuleData> {
    fun getFactory(formatData: T, tag: String?): TransformFactory
}
```

Here `tag` is an arbitrary `String` that that has been attached to the node allowing us to inject different
implementations for the same node type into the graph. It is optional and thus may be `null`.

`TransformFactory` is:

```kotlin
interface TransformFactory {
  fun createNode(transformData: InputStream, sources: List<GraphNode>): DataTransform
}
```

where `transformData` is the data in the file for that node. The contents of that file need only make sense for the
`TransformFactory`.

Note that often a node contains the `ValueToken`s of the inputs it will process. However, it should not serialize the
full token as that may not be valid if the producing node's implementation changes. Instead it should serialize just
the `ValueId`. The `sources` array passed to `createNode` can be used to convert `ValueId`s into `ValueToken`s.

NOTE: This means we have to be sure to serialize (or all least construct things) in topological order.

# Injection

Each `FormatModule` manages its own injection. That is because different formats must be deserialized and injected
differently. All of the `anxml` nodes use the same format: `StandardJson`. 

## StandardJson Injection

The `StandardJsonFormatModule` allows you to register `TransformFactory` implementations for tags or classes. When
`getFactory` is called the module searches for `TransformFactory` implementations in the folowing order:

1. By tag: if there is a `TransformFactory` for the specified tag it is returned.
2. By class: If the class specified in the `StandardJsonFormatModuleData` has had a `TransformFactory` registered it is
   returned. 
3. The stadard `TransformFactory` is used. This does a little work to turn the `ValueId`s into `ValueToken`s and then
   allows [Jackson](https://github.com/FasterXML/jackson) to handle the rest of the deserialization.
   
Note that this is compatible with other injection frameworks. For example, if using
[Guice](https://github.com/google/guice) you might create an `@Provides` annotated `TransformFactory` that has things
like database connections injected into it allowing it to access both injection components and the serialized
representation.

## Injection Notes
 
Note that this framework does not enforce any class hierarchy. A `TransformFactory` can return any `DataTransform` that
is compatible with the graph even if the implementation returned does not share any interface or base class with the
serialized version (other than both being `DataTransform`).