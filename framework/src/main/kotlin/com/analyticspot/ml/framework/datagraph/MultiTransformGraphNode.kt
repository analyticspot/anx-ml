package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.MultiTransform
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReferenceArray

/**
 *
 */
internal class MultiTransformGraphNode protected constructor(builder: Builder) : GraphNode(builder) {
    val transform: MultiTransform = builder.transform ?: throw IllegalArgumentException("Transform must be non-null")

    companion object {
        private val log = LoggerFactory.getLogger(Companion::class.java)

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

        fun build(): MultiTransformGraphNode = MultiTransformGraphNode(this)
    }

    private class ExecutionManager(override val graphNode: MultiTransformGraphNode, private val parent: GraphExecution)
        : NodeExecutionManager {

        private val dataSets = AtomicReferenceArray<DataSet?>(graphNode.sources.size)
        private val numReceived = AtomicInteger(0)

        override fun onDataAvailable(subId: Int, data: DataSet) {
            log.debug("Node {} notified that data with source index {} is available", graphNode.id, subId)
            assert(subId >= 0 && subId < graphNode.sources.size) {
                "Source index had illegal value $subId"
            }
            check(dataSets.getAndSet(subId, data) == null) {
                "Data for index $subId was already received."
            }
            val received = numReceived.incrementAndGet()
            log.debug("Node {} has now received {} of {} data sets", graphNode.id, received, graphNode.sources.size)
            if (received == graphNode.sources.size) {
                log.debug("Node {} has received all data sets. Registering with parent to run.", graphNode.id)
                parent.onReadyToRun(this)
            }
        }

        override fun run(): CompletableFuture<DataSet> {
            log.debug("Node {} is executing.", graphNode.id)
            val sourceList: List<DataSet> = graphNode.sources.indices.map {
                dataSets.get(it) ?: throw IllegalStateException("Data set for index $it was missing")
            }
            return graphNode.transform.transform(sourceList).whenComplete { dataSet, throwable ->
                // Get rid of the references to all sources so the memory can be GC'd
                graphNode.sources.indices.forEach { idx -> dataSets.set(idx, null) }
            }
        }

    }
}
