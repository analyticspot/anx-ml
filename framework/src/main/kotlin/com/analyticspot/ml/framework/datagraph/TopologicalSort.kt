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
    val sorter = TopoSorter(graph, { node -> node.subscribers })
    return sorter.sort()
}

/**
 * Returns a forward topological sort of the [DataGraph] that **does** include training-only nodes. This guarantees
 * that the returned `Iterable` will return all nodes that send data to node `X` before node `X` is returned (including
 * when the data is sent to `X` only during training).
 */
fun sortWithTrain(graph: DataGraph): Iterable<GraphNode> {
    val sorter = TopoSorter(graph, { node -> node.subscribers.plus(node.trainOnlySubscribers) })
    return sorter.sort()
}

/**
 * Returns a "backward" topological sort of the [DataGraph] that does **not** include training-only nodes. This
 * guarantees that the returned `Iterable` will return a node `X` before any node that sends data to `X`.
 */
fun sortBackwards(graph: DataGraph): Iterable<GraphNode> {
    val sorter = TopoSorter(graph, { node -> node.sources })
    return sorter.sort()
}

/**
 * Returns a "backward" topological sort of the [DataGraph] that **does** include training-only nodes. This
 * guarantees that the returned `Iterable` will return a node `X` before any node that sends data to `X` (including
 * when the data is sent to `X` only during training).
 */
fun sortWithTrainBackwards(graph: DataGraph): Iterable<GraphNode> {
    val sorter = TopoSorter(graph, { node -> node.sources.plus(node.trainOnlySources) })
    return sorter.sort()
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

    fun sort(): List<GraphNode> {
        for (node in graph.allNodes) {
            visit(node)
        }
        assert(result.size == graph.allNodes.size)
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

