package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.SingleObservationDataSet
import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.observation.ArrayObservation
import com.analyticspot.ml.framework.observation.Observation
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * A [DataGraph] is a directed acyclic graph of [GraphNode] objects. Typically each [GraphNode] represents a
 * [DataTransform]. Thus the graph edges are data sources and the nodes are transformations. The [source] [GraphNode]
 * is the root of the [DataGraph] and it where the original source data that is used for learning or prediciton comes
 * from. The [result] [GraphNode] is the final node; the value it produces is the final output of the entire graph. For
 * classification or regression problems the output of the [result] contain the prediction (and, perhaps, some
 * metadata).
 */
class DataGraph(builder: GraphBuilder) {
    val source: GraphNode
    val result: GraphNode
    // An array of all the GraphNodes such that a node `x` can be found at `allNodes[x.id]`.
    internal val allNodes: Array<GraphNode>

    init {
        source = builder.source
        result = builder.result

        allNodes = builder.nodesById.toTypedArray()
    }

    companion object {
        /**
         * Kotlin-style builder for a [DataGraph].
         */
        fun build(init: GraphBuilder.() -> Unit): DataGraph {
            with(GraphBuilder()) {
                init()
                return build()
            }
        }
    }

    /**
     * Constructs an [Observation] that is compatible with the types/tokens specified for [source].
     */
    fun buildTransformSource(vararg values: Any): Observation {
        val baseArray = Array<Any>(values.size) { idx ->
            check(values[idx].javaClass == source.tokens[idx].clazz)
            values[idx]
        }
        return ArrayObservation(baseArray)
    }

    /**
     * Run the data through the entire graph. The result type is a future of `DataSet` because the graph
     * might contain an asynchronous [DataTransform] or it might contain an [OnDemandValue].
     */
    fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val graphExec = GraphExecution(this, ExecutionType.TRANSFORM, exec)
        return graphExec.execute(dataSet)
    }

    /**
     * Convenience overload that transforms a single [Observation].
     */
    fun transform(observation: Observation, exec: ExecutorService): CompletableFuture<Observation> {
        return transform(SingleObservationDataSet(observation), exec).thenApply {
            it.first()
        }
    }

    class GraphBuilder {
        internal lateinit var source: SourceGraphNode

        lateinit var result: GraphNode

        // An array of GraphNode such that nodesById[idx] returns the GraphNode whose id is idx.
        internal val nodesById: MutableList<GraphNode> = mutableListOf()

        internal var nextId = 0

        fun setSource(init: SourceGraphNode.Builder.() -> Unit): GraphNode {
            source = SourceGraphNode.build(nextId++, init)
            assert(nodesById.size == source.id)
            nodesById.add(source)
            return source
        }

        fun addTransform(src: GraphNode, transform: DataTransform): GraphNode {
            val node = TransformGraphNode.build(nextId++) {
                this.transform = transform
                sources += src
            }
            src.subscribers += node
            assert(nodesById.size == node.id)
            nodesById.add(node)
            return node
        }

        fun build(): DataGraph {
            return DataGraph(this)
        }
    }
}
