package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.LearningTransform
import org.slf4j.LoggerFactory

/**
 * A [GraphNode] which takes a single input [DataSet] and applies a [LearningTransform] to it.
 */
class LearningGraphNode(builder: Builder) : GraphNode(builder) {
    val transform: LearningTransform = builder.transform ?: throw IllegalArgumentException("Transform must be non-null")

    companion object {
        fun build(id: Int, init: Builder.() -> Unit): LearningGraphNode {
            return with(Builder(id)) {
                init()
                build()
            }
        }
    }

    override fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager =
            ExecutionManager(this, execType, parent)

    class Builder(id: Int) : GraphNode.Builder(id) {
        var transform: LearningTransform? = null
            set(value) {
                field = value ?: throw IllegalArgumentException("Transform can not be null")
                tokens.addAll(value.description.tokens)
                trainOnlyTokens.addAll(value.description.trainOnlyTokens)
                tokenGroups.addAll(value.description.tokenGroups)
            }

        override fun build(): LearningGraphNode = LearningGraphNode(this)
    }

    // The execution manager for this node. Since this expects only a single input it signals onReadyToRun as soon as
    // onDataAvailable is called.
    private class ExecutionManager(
            override val graphNode: LearningGraphNode,
            private val execType: ExecutionType,
            parent: GraphExecution) : SingleInputExecutionManager(parent) {

        companion object {
            private val log = LoggerFactory.getLogger(ExecutionManager::class.java)
        }

        override fun doRun(dataSet: DataSet): DataSet {
            if (execType == ExecutionType.TRANSFORM) {
                return graphNode.transform.transform(dataSet)
            } else {
                check(execType == ExecutionType.TRAIN_TRANSFORM)
                return graphNode.transform.trainTransform(dataSet)
            }
        }
    }
}
