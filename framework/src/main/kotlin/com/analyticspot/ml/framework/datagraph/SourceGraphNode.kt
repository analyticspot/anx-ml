package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.Observation
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 *
 */
class SourceGraphNode private constructor(builder: GraphNode.Builder) : GraphNode(builder) {
    companion object {
        private val log = LoggerFactory.getLogger(SourceGraphNode::class.java)
    }

    override fun transformWithSource(graphSource: Observation, exec: ExecutorService): CompletableFuture<Observation> {
        log.debug("transformWithSource called on source node. Returning the source.")
        return CompletableFuture.completedFuture(graphSource)
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
}
