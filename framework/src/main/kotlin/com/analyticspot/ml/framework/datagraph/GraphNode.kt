package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.DataDescription

/**
 * This is the base class for all [GraphNode]s. Each such node represents a single node in the graph. It holds the
 * metadata about that node (what its inputs are, what its output is, how it transforms its input into its output,
 * etc.).
 */
abstract class GraphNode internal constructor(builder: Builder) : DataDescription(builder) {
    internal val sources: List<GraphNode> = builder.sources
    internal val subscribers: MutableList<GraphNode> = mutableListOf()
    internal val trainOnlySubscribers: MutableList<GraphNode> = mutableListOf()
    internal val id: Int = builder.id

    /**
     * Return a [NodeExecutionManager] for the given operation (`train`, `trainTransform`, or `execute`).
     */
    abstract fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager

    open class Builder(internal val id: Int) : DataDescription.Builder() {
        val sources: MutableList<GraphNode> = mutableListOf()
    }
}

