# Overview

Given a `DataGraph` we want to run data through the entire graph to learn or execute. We could do this in two
different ways:

1. Have the result `GraphNode` "pull" its source data sets an then compute itself. The `GraphNode`s the result pulled
   would also pull their sources, etc. until we go all the way to the source node.
2. Have the source `GraphNode` "push" it's result data to all nodes that are "subscribed" to it. Those nodes then
   push their outputs to any subscribers, etc. until we end up at the result node.

# Pull

While it may seem less natural, the naive implementation of the pull technique is easier to implement and seems to have
several advantages.

The first cut at such an algorithm might look something like:

```kotlin
class GraphNode {
    fun execute(): CompletableFuture<Observation> {
        // First request all the data from all required sources
        val requiredSources: MutableList<CompletableFuture<Observation>> = mutableListOf()
        for (src in sources) {
            requiredSources += src.execute()
        }
        // Once it's all ready, compute the underlying transformation
        return CompletableFuture.allOf(requiresSources).thenApply {
            val listOfObservations = requiredSources.map { it.get() }
            return transformation.execute(listOfObservations)
        }
    }
}
```

This has a few advantages:

* The algorithm is pretty straight forward: each node simply requests all it's input and, once they're available it
  computes and returns its transformation.
* Data that is no longer needed becomes eligable for garbage collection: Supposed we have a node `A` that is consumed by
  both nodes `B` and `C`. `D`, our result, then consumes `C`. Note that as soon as `B` and `C` have computed their
  output nothing is holding a reference to `A`'s output anymore. This isn't such a big deal for the `execute` method
  since that's operating on just a single observation, but `train` and `trainTransform` which operate on whole data sets
  and can involve memory intensive algorithms, can generally benefit quite a bit from the ability to free memory like
  this; particularly with complex graphs.
* A mix of synchronous and asynchronous algorithms works naturally: everything just returns a future; some `GraphNode`
  subclasses compute their result synchronously and return an already complete future while others wrap an asynchronous
  `DataTransformation` and so they simply return the `CompletableFuture` returned by that transformation.
  
However, there are a few problems with this approach:

## Where's the Source

Since the data is pulled from the `result` node when we hit the `source` node it won't know what its value is. There's
two options here neither of which is very nice:

1. Call something like `DataGraph.setSource` and then call `execute`. The problem here is that the `DataGraph` is then
   stateful so we can't run two different observations through the same graph in two different threads. It is also
   not re-entrant so becomes more error prone.
2. Change the signature of `execute` to something like `execute(sourceObservation: Observation)`. The
   `sourceObservation` is then passed backward through the graph until we hit the source `GraphNode`. The execute
   method of that node simply returns the data passed to it. This is re-entrant but the API is unnatural.
   
## Double Executions

Consider a graph like the following:

```
     source
       |
       A
     /  \
    B   C
    \  /
    result
```

`result` will try to pull it's data from `B` and `C`. They will then **each** request their data from `A` which will
cause `A` to compute it's result twice. Not very efficient.

Fixing this problem is complicated by the fact that we want to be able to call `execute` on the same graph in
multiple different threads simultaneously. We therefore need some kind of cache that is unique for each call to
`execute` which means we either pass this cache up through the graph or we introduct an additionaly component.

## Scheduling

The naive algorithm simply called `tranformation.execute(observation)` on the current thread. This doesn't allow us
use all the CPU cores or control the thread used for each computation. We could include an `ExecutorService` parameter
in the `GraphNode.execute` method and pass that up the chain. That works but now each subclass of `GraphNode` must
remember to properly schedule it's work.

# Push

Push is a bit more complicated to implement but solves all of the above problems. The algorithm is as follows:

* Each `GraphNode` has a `getExecutionManager` method which returns an `ExecutionManager` instance.
* The `ExecutionManager` doesn't do a lot: it keeps references to the data needed by the underlying transformation until
  all the required data is available. It then signals a `GraphExection` instance that it's ready to execute.
* The `GraphExecution` instance, which is unique to each `execute` call, then handles all scheduling, etc.

See the `GraphExecution` and `ExecutionManager` classes for details.


