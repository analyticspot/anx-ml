package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.ValueToken

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
        val tokens = mutableListOf<ValueToken<*>>()
        val trainOnlyTokens = mutableListOf<ValueToken<*>>()

        fun build(): SourceGraphNode {
            val gnb = GraphNode.Builder(id).apply {
                tokens.addAll(tokens)
                trainOnlyTokens.addAll(trainOnlyTokens)
            }
            return SourceGraphNode(gnb)
        }
    }

    override fun getExecutionManager(parent: GraphExecution): NodeExecutionManager {
        throw IllegalStateException("This is a SourceGraphNode and it therefore does not participate in the normal " +
            "GraphExecution protocol.")
    }
}
