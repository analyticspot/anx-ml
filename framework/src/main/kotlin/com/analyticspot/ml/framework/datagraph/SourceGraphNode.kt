package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.ColumnId

/**
 * A source when we know exactly what columns are required for the source node. See [SourceGraphNodeBase] for details.
 */
class SourceGraphNode private constructor(builder: Builder) : SourceGraphNodeBase(builder.gnBuilder) {
    val trainOnlyColumnIds = builder.trainOnlyColumnIds
    /**
     * This is **all** columnIds, **including** the trainOnlyColumnIds
     */
    val columnIds = builder.columnIds.plus(trainOnlyColumnIds)

    companion object {
        /**
         * Construct a [SourceGraphNode].
         *
         * @param id the unique id of this node in the [DataGraph]. The [DataGraph.Builder] is generally responsible for
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
        val columnIds = mutableListOf<ColumnId<*>>()

        /**
         * Described the values in the source data set which are only required when training. A typical example would be
         * the training target for a supervised learning algorithm.
         */
        val trainOnlyColumnIds = mutableListOf<ColumnId<*>>()

        internal lateinit var gnBuilder: GraphNode.Builder

        fun build(): SourceGraphNode {
            gnBuilder = GraphNode.Builder(id)
            return SourceGraphNode(this)
        }
    }
}
