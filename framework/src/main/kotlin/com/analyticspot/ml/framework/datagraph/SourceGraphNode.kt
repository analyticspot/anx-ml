package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId

/**
 * A special [GraphNode] for the one node that is the source for the entire graph. While most nodes know how to compute
 * their outputs given their inputs this node has no inputs and we don't want it to be too knowledgeable: if it knew
 * how to compute its own outputs it would be hard to do one run on data read from a file, the next run on data in RAM,
 * and the next run on data from a database. Instead the source node is really nothing more than a description of the
 * data that is required by the rest of teh graph. To generate a concrete instance of this backed by data one would use
 * something like [DataGraph.buildTransformSource].
 */
class SourceGraphNode private constructor(builder: GraphNode.Builder) : GraphNode(builder) {

    companion object {
        /**
         * Construct a [SourceGraphNode].
         *
         * @param id the unique id of this node in the [DataGraph]. The [DataGraph.Builder] is gnerally responsible for
         * generating suitable ids.
         * @param init a lambda to be executed with a [DataGraph.Builder] as the receiver This configures the source.
         */
        fun build(id: Int, init: Builder.() -> Unit): SourceGraphNode {
            return with(Builder(id)) {
                init()
                build()
            }
        }
    }

    class Builder(private val id: Int) {
        /**
         * Described the values in the source data set.
         */
        val valueIds = mutableListOf<ValueId<*>>()

        /**
         * Described the values in the source data set which are only required when training. A typical example would be
         * the training target for a supervised learning algorithm.
         */
        val trainOnlyValueIds = mutableListOf<ValueId<*>>()

        fun build(): SourceGraphNode {
            var arrayIdx = 0
            val gnb = GraphNode.Builder(id).apply {
                tokens.addAll(valueIds.map {
                    IndexValueToken.create(arrayIdx++, it)
                })
                trainOnlyTokens.addAll(trainOnlyValueIds.map {
                    IndexValueToken.create(arrayIdx++, it)
                })
            }
            return SourceGraphNode(gnb)
        }
    }

    override fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager {
        return ExecutionManager(this)
    }

    // Note that source data nodes are specicial and don't really participate in the GraphExecution protocol. Thus all
    // methods here just throw exceptions.
    private class ExecutionManager(override val graphNode: GraphNode) : NodeExecutionManager {
        override fun onDataAvailable(data: DataSet) {
            throw IllegalStateException("This is a SourceGraphNode and it therefore does not participate in the " +
                    "normal GraphExecution protocol.")
        }

        override fun run() {
            throw IllegalStateException("This is a SourceGraphNode and it therefore does not participate in the " +
                    "normal GraphExecution protocol.")
        }
    }
}
