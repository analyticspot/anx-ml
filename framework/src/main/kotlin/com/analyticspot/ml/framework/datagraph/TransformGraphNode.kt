package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import org.slf4j.LoggerFactory

/**
 * A [GraphNode] that takes a single input, runs it through a [DataTransform] and produces a single output.
 */
internal open class TransformGraphNode protected constructor(builder: Builder)
    : HasTransformGraphNode<SingleDataTransform>(builder) {
    override val transform: SingleDataTransform = builder.transform ?:
            throw IllegalArgumentException("Transform can not be null")

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

    override fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager =
            ExecutionManager(this, parent)

    open class Builder(id: Int) : GraphNode.Builder(id) {
        var transform: SingleDataTransform? = null
            set(value) {
                field = value ?: throw IllegalArgumentException("Transform can not be null")
                tokens.addAll(value.description.tokens)
                tokenGroups.addAll(value.description.tokenGroups)
                check(value.description.trainOnlyTokens.size == 0) {
                    "Non-trained DataTranform should not declare any train-only tokens"
                }
            }

        override fun build(): TransformGraphNode = TransformGraphNode(this)
    }

    // The execution manager for this node. Since this expects only a single input it signals onReadyToRun as soon as
    // onDataAvailable is called.
    private class ExecutionManager(override val graphNode: TransformGraphNode, parent: GraphExecution)
        : SingleInputExecutionManager(parent) {

        override fun doRun(dataSet: DataSet) = graphNode.transform.transform(dataSet)
    }
}

