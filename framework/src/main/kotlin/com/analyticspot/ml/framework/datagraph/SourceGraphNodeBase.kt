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
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * A special [GraphNode] for the one node that is the source for the entire graph. While most nodes know how to compute
 * their outputs given their inputs this node has no inputs and we don't want it to be too knowledgeable: if it knew
 * how to compute its own outputs it would be hard to do one run on data read from a file, the next run on data in RAM,
 * and the next run on data from a database. There are two subclasses of [SourceGraphNodeBase]:
 *
 * * [DataSetSourceGraphNode] this source knows nothing about the columns that are required to be in the source - it
 *   simply takes an input [DataSet] and passes that through the graph. This is useful when decomposing a [DataGraph]
 *   into multiple sub-graphs; particularly when the set of columns produced by the first graph isn't known until it
 *   has been trained. Note that you can't call [DataGraph.buildSourceObservation] if the source is a
 *   [DataSetSourceGraphNode].
 * * [SourceGraphNode] this source is really nothing more than a description of the data that is required by the rest
 *   of the graph. To generate a concrete instance of this backed by data one would use something like
 *   [DataGraph.buildSourceObservation].
 */
abstract class SourceGraphNodeBase protected constructor(builder: GraphNode.Builder) : GraphNode(builder) {
    override fun getExecutionManager(parent: GraphExecutionProtocol, execType: ExecutionType): NodeExecutionManager {
        return ExecutionManager(this)
    }

    // Note that source data nodes are special and don't really participate in the GraphExecution protocol. Thus all
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
