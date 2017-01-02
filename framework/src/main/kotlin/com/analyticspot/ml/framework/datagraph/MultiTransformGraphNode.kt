package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.MultiTransform
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReferenceArray

/**
 *
 */
internal class MultiTransformGraphNode protected constructor(builder: Builder) : GraphNode(builder) {
    val transform: MultiTransform = builder.transform ?: throw IllegalArgumentException("Transform must be non-null")

    companion object {
        fun build(id: Int, init: Builder.() -> Unit): MultiTransformGraphNode {
            return with(Builder(id)) {
                init()
                build()
            }
        }
    }

    override fun getExecutionManager(parent: GraphExecution, execType: ExecutionType): NodeExecutionManager {
        return ExecutionManager(this, parent)
    }

    class Builder(id: Int) : GraphNode.Builder(id) {
        var transform: MultiTransform? = null
            set(value) {
                field = value ?: throw IllegalArgumentException("Transform can not be null")
                tokens.addAll(value.description.tokens)
                tokenGroups.addAll(value.description.tokenGroups)
                check(value.description.trainOnlyTokens.size == 0) {
                    "Non-trained DataTranform should not declare any train-only tokens"
                }
            }

        override fun build(): MultiTransformGraphNode = MultiTransformGraphNode(this)
    }

    private class ExecutionManager(override val graphNode: MultiTransformGraphNode, private val parent: GraphExecution)
        : NodeExecutionManager {

        private val dataSets = AtomicReferenceArray<DataSet?>(graphNode.sources.size)
        private val numReceived = AtomicInteger(0)

        override fun onDataAvailable(sourceIdx: Int, data: DataSet) {
            check(dataSets.getAndSet(sourceIdx, data) == null) {
                "Data for index $sourceIdx was already received."
            }
            val received = numReceived.incrementAndGet()
            if (received == graphNode.sources.size) {
                parent.onReadyToRun(this)
            }
        }

        override fun run() {
            val sourceList: List<DataSet> = graphNode.sources.indices.map {
                dataSets.get(it) ?: throw IllegalStateException("Data set for index $it was missing")
            }
            val resultF = graphNode.transform.transform(sourceList)
            resultF.thenAccept {
                // Get rid of the references to all sources so the memory can be GC'd
                graphNode.sources.indices.forEach { idx -> dataSets.set(idx, null) }
                parent.onDataComputed(this, it)
            }
        }

    }
}
