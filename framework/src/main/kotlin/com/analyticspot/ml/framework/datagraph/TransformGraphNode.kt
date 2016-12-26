package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.observation.Observation
import org.slf4j.LoggerFactory

/**
 *
 */
internal open class TransformGraphNode protected constructor(builder: Builder) : GraphNode(builder) {
    val transform: DataTransform = builder.transform ?: throw IllegalArgumentException("Transform can not be null")

    companion object {
        private val log = LoggerFactory.getLogger(Companion::class.java)
        fun build(id: Int, init: Builder.() -> Unit): TransformGraphNode {
            return with(Builder(id)) {
                init()
                build()
            }
        }
    }

    open class Builder(id: Int) : GraphNode.Builder(id) {
        var transform: DataTransform? = null
            set(value) {
                field = value ?: throw IllegalArgumentException("Transform can not be null")
                tokens.addAll(value.description.tokens)
                tokenGroups.addAll(value.description.tokenGroups)
            }

        override fun build(): TransformGraphNode = TransformGraphNode(this)
    }

    override fun getExecutionManager(parent: GraphExecution): NodeExecutionManager = ExecutionManager(this, parent)

    private class ExecutionManager(override val graphNode: TransformGraphNode, private val parent: GraphExecution)
        : NodeExecutionManager {
        @Volatile
        private var observation: Observation? = null

        override fun onDataAvailable(observation: Observation) {
            this.observation = observation
            parent.onReadyToRun(this)

        }

        override fun run() {
            val result = graphNode.transform.transform(observation!!)
            // Get rid of our reference to the observaton so it can be GC'd if nothing else is using it.
            observation = null
            parent.onDataComputed(this, result)
        }
    }
}

