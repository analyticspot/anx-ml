package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.observation.ArrayObservation
import com.analyticspot.ml.framework.observation.Observation
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 *
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
     * Run the observation through the entire graph. The result type is a future of `Observation` because the graph
     * might contain an asynchronous [DataTransform] or it might contain an [OnDemandValue].
     */
    fun transform(observation: Observation, exec: ExecutorService): CompletableFuture<Observation> {
        val graphExec = GraphExecution(this, exec)
        return graphExec.transform(observation)
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
