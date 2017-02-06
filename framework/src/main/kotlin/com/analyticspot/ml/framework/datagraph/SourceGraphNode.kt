/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * The ANX ML library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * A special [GraphNode] for the one node that is the source for the entire graph. While most nodes know how to compute
 * their outputs given their inputs this node has no inputs and we don't want it to be too knowledgeable: if it knew
 * how to compute its own outputs it would be hard to do one run on data read from a file, the next run on data in RAM,
 * and the next run on data from a database. Instead the source node is really nothing more than a description of the
 * data that is required by the rest of teh graph. To generate a concrete instance of this backed by data one would use
 * something like [DataGraph.buildSourceObservation].
 */
class SourceGraphNode private constructor(builder: Builder) : GraphNode(builder.gnBuilder) {
    internal val trainOnlyColumnIds = builder.trainOnlyColumnIds
    override val transformDescription: TransformDescription = TransformDescription(
            builder.columnIds + builder.trainOnlyColumnIds)

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

    override fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager {
        return ExecutionManager(this)
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

    // Note that source data nodes are specicial and don't really participate in the GraphExecution protocol. Thus all
    // methods here just throw exceptions.
    private class ExecutionManager(override val graphNode: GraphNode) : NodeExecutionManager {
        override fun onDataAvailable(subId: Int, data: DataSet) {
            throw IllegalStateException("This is a SourceGraphNode and it therefore does not participate in the " +
                    "normal GraphExecution protocol.")
        }

        override fun run(exec: ExecutorService): CompletableFuture<DataSet> {
            throw IllegalStateException("This is a SourceGraphNode and it therefore does not participate in the " +
                    "normal GraphExecution protocol.")
        }
    }
}
