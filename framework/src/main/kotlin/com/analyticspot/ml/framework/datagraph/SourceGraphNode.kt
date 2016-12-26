package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.observation.Observation

/**
 *
 */
class SourceGraphNode private constructor(builder: GraphNode.Builder) : GraphNode(builder) {

    companion object {
        fun build(id: Int, init: Builder.() -> Unit): SourceGraphNode {
            return with(Builder(id)) {
                init()
                build()
            }
        }
    }

    class Builder(private val id: Int) {
        val tokens = mutableListOf<IndexValueToken<*>>()
        val trainOnlyTokens = mutableListOf<IndexValueToken<*>>()

        fun build(): SourceGraphNode {
            val gnb = GraphNode.Builder(id).apply {
                tokens.addAll(this@Builder.tokens)
                trainOnlyTokens.addAll(this@Builder.trainOnlyTokens)
            }
            return SourceGraphNode(gnb)
        }
    }

    override fun getExecutionManager(parent: GraphExecution): NodeExecutionManager = ExecutionManager(this)

    private class ExecutionManager(override val graphNode: GraphNode) : NodeExecutionManager {
        override fun onDataAvailable(observation: Observation) {
            throw IllegalStateException("This is a SourceGraphNode and it therefore does not participate in the " +
                    "normal GraphExecution protocol.")
        }

        override fun run() {
            throw IllegalStateException("This is a SourceGraphNode and it therefore does not participate in the " +
                    "normal GraphExecution protocol.")
        }

    }
}
