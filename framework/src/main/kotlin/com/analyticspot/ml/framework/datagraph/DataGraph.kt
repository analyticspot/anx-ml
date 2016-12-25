package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.observation.Observation
import java.util.ArrayList
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 *
 */
class DataGraph(builder: GraphBuilder) {
    val source: GraphNode
    val result: GraphNode
    // An array of all the GraphNodes such that a node `x` can be found at `allNodes[x.id]`.
    private val allNodes: Array<GraphNode>

    init {
        source = builder.source
        result = builder.result

        fun recursiveFillNodes(nodes: ArrayList<GraphNode>, node: GraphNode) {
            nodes[node.id] = node
            node.subscribers.forEach { recursiveFillNodes(nodes, it) }
            node.trainOnlySubscribers.forEach { recursiveFillNodes(nodes, it) }
        }
        val nodes = ArrayList<GraphNode>(builder.nextId)
        recursiveFillNodes(nodes, source)
        allNodes = nodes.toTypedArray()
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
     * Run the observation through the entire graph. The result type is a future of `Observation` because the graph
     * might contain an asynchronous [DataTransform] or it might contain an [OnDemandValue].
     */
    fun transform(observation: Observation, exec: ExecutorService): CompletableFuture<Observation> {
        return result.transformWithSource(observation, exec)
    }

    class GraphBuilder {
        lateinit var source: SourceGraphNode

        lateinit var result: GraphNode

        internal var nextId = 0

        fun setSource() {

        }

        fun addTransform(src: GraphNode, transform: DataTransform): GraphNode {
            return TransformGraphNode.build(nextId++) {
                this.transform = transform
                sources += src
            }
        }

        fun build(): DataGraph {
            return DataGraph(this)
        }
    }
}
