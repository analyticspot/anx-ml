package com.analyticspot.ml.framework.datagraph

import java.util.LinkedList

/**
 * Iterates over the [GraphNode] instances in a [DataGraph] in topological order so that the first item returned is the
 * graph's source, the following items are returned only after all the sources they use have been returned, and the
 * graph's result is returned last.
 */
class TopologicalIterator(private val graph: DataGraph) : Iterator<GraphNode> {
    // missingSources[i] is the number of sources that we have not yet produced that are required by the GraphNode with
    // id i.
    private val missingSources: Array<Int> = Array(graph.allNodes.size) { idx ->
        graph.allNodes[idx].sources.size
    }

    private val readyNodes = LinkedList<GraphNode>()

    init {
        readyNodes.push(graph.source)
    }

    override fun hasNext(): Boolean {
        return readyNodes.size > 0
    }

    override fun next(): GraphNode {
        val toReturn = readyNodes.pop()
        for (s in toReturn.subscribers) {
            missingSources[s.id] -= 1
            if (missingSources[s.id] <= 0) {
                assert(missingSources[s.id] == 0)
                readyNodes.push(s)
            }
        }
        return toReturn
    }
}
