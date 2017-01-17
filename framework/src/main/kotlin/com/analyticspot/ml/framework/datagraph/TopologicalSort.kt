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

import java.util.LinkedList

/**
 * Functions for performing topological sorts.
 */

/**
 * Returns a forward topological sort of the [DataGraph] that does **not** include training-only nodes. This guarantees
 * that the returned `Iterable` will return all nodes that send data to node `X` before node `X` is returned.
 */
fun sort(graph: DataGraph): Iterable<GraphNode> {
    val sorter = TopoSorter(graph, { node -> node.subscribers.map { it.subscriber } })
    return sorter.sort(graph.source)
}

/**
 * Returns a forward topological sort of the [DataGraph] that **does** include training-only nodes. This guarantees
 * that the returned `Iterable` will return all nodes that send data to node `X` before node `X` is returned (including
 * when the data is sent to `X` only during training).
 */
fun sortWithTrain(graph: DataGraph): Iterable<GraphNode> {
    val sorter = TopoSorter(graph, { node -> node.subscribers.map { it.subscriber }
            .plus(node.trainOnlySubscribers.map { it.subscriber }) })
    return sorter.sort(graph.source)
}

/**
 * Returns a "backward" topological sort of the [DataGraph] that does **not** include training-only nodes. This
 * guarantees that the returned `Iterable` will return a node `X` before any node that sends data to `X`.
 */
fun sortBackwards(graph: DataGraph): Iterable<GraphNode> {
    val sorter = TopoSorter(graph, { node -> node.sources.map { it.source } })
    return sorter.sort(graph.result)
}

/**
 * Returns a "backward" topological sort of the [DataGraph] that **does** include training-only nodes. This
 * guarantees that the returned `Iterable` will return a node `X` before any node that sends data to `X` (including
 * when the data is sent to `X` only during training).
 */
fun sortWithTrainBackwards(graph: DataGraph): Iterable<GraphNode> {
    val sorter = TopoSorter(graph, { node -> node.sources.map { it.source }
            .plus(node.trainOnlySources.map { it.source }) })
    return sorter.sort(graph.result)
}

/**
 * Performs a depth-first topological sort as per Tarjan's algorithm: https://en.wikipedia.org/wiki/Topological_sorting.
 * The only difference between forward, backward, with and without training data is which nodes count as connected.
 * Thus, we pass that in.
 *
 * Note that one reason we choose this algorithm is that it relies on only one specification of the graph: either
 * sources or subscribers, but not both. This allows us to build a graph by specifying only sources and then later
 * determine the subscriber settings from the sources.
 */
private class TopoSorter(private val graph: DataGraph,
        private val getConnectedNodes: (GraphNode) -> Iterable<GraphNode>) {
    private val tempMarked = Array<Boolean>(graph.allNodes.size) { false }
    private val marked = Array<Boolean>(graph.allNodes.size) { false }
    private val result = LinkedList<GraphNode>()

    /**
     * We know that all nodes (at least all nodes we care about) are accessible via source (if we're sorting forwards)
     * or the result (if we're sorting backwards). Thus, unlike Tarjan's "full" algorithm we don't start with a list of
     * all nodes, we start with just one node. This also ensures that train-only nodes don't end up as part of the
     * sort when we're doing a regular, non-training, sort.
     */
    fun sort(startNode: GraphNode): List<GraphNode> {
        visit(startNode)
        return result
    }

    fun visit(node: GraphNode) {
        if (tempMarked[node.id]) {
            throw IllegalStateException("This graph contains a cycle.")
        }
        if (!marked[node.id]) {
            tempMarked[node.id] = true
            getConnectedNodes(node).forEach { visit(it) }
            marked[node.id] = true
            tempMarked[node.id] = false
            result.addFirst(node)
        }
    }
}

