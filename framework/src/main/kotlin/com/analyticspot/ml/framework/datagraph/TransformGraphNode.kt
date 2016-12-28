package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.DataTransform
import org.slf4j.LoggerFactory

/**
 * A [GraphNode] that takes a single input, runs it through a [DataTransform] and produces a single output.
 */
internal open class TransformGraphNode protected constructor(builder: Builder) : GraphNode(builder) {
    val transform: DataTransform = builder.transform ?: throw IllegalArgumentException("Transform can not be null")

    companion object {
        private val log = LoggerFactory.getLogger(Companion::class.java)

        /**
         * Construct a [TransformGraphNode] by using the Kotlin builder pattern.
         */
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

    override fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager =
        ExecutionManager(this, parent)

    // The execution manager for this node. Since this expects only a single input it signals onReadyToRun as soon as
    // onDataAvailable is called.
    private class ExecutionManager(override val graphNode: TransformGraphNode, private val parent: GraphExecution)
        : NodeExecutionManager {
        @Volatile
        private var data: DataSet? = null

        override fun onDataAvailable(data: DataSet) {
            this.data = data
            parent.onReadyToRun(this)

        }

        override fun run() {
            val result = graphNode.transform.transform(data!!)
            // Get rid of our reference to the observaton so it can be GC'd if nothing else is using it.
            data = null
            parent.onDataComputed(this, result)
        }
    }
}

