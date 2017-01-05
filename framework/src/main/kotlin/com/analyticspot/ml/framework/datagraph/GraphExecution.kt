package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Manages the execution of a single observation through the graph. See the PUSH.OR.PULL.README.md file in this
 * directory for details. It manages the execution by cooperating with a set of [NodeExecutionManager] instances: one
 * per node in the graph. These [NodeExecutionManager] instances keep track of what data is needed by the node they
 * manage and letting the [GraphExecution] know when the necessary data is available. Similarly, they also take
 * care of actually running the node and signaling back to the [GraphExecution] when the run is complete. This latter
 * level of indirection allows us to push the details of synchronous vs. asynchronous execution to the
 * [NodeExecutionManager] rather than having the [GraphExecution] manage it.
 *
 *
 * Protocol:
 *
 * * When data is avaiable the [NodeExecutionManager.onDataAvailable] method will be called for all nodes that have
 *   subscribed to that data.
 * * When a [NodeExecutionManager] has all the data it needs to run that manager will call [onReadyToRun] on this
 *   instance. This will then schedule the node to be run (often on another thread).
 * * Once scheduled the [NodeExecutionManager.run] method wil be called. The [NodeExecutionManager] will then call its
 *   underlying [DataTransform] to generate the data.
 * * Once the run is complete and the data is available the future returned by [NodeExecutionManager.run] will complete
 *   causing any subscribing nodes to be notified.
 */
class GraphExecution (
        private val dataGraph: DataGraph, private val execType: ExecutionType, private val exec: ExecutorService) {
    private val executionManagers: Array<NodeExecutionManager>
    private val executionResult: CompletableFuture<DataSet> = CompletableFuture()

    init {
        val graphNodes = dataGraph.allNodes
        executionManagers = Array<NodeExecutionManager>(graphNodes.size) { idx ->
            graphNodes[idx].getExecutionManager(this, execType)
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(GraphExecution::class.java)
    }

    fun execute(data: DataSet): CompletableFuture<DataSet> {
        log.debug("Starting execution of DataGraph")
        onDataComputed(executionManagers[dataGraph.source.id], data)
        return executionResult
    }

    fun onDataComputed(manager: NodeExecutionManager, data: DataSet) {
        log.debug("Node {} has finished computing its Observation.", manager.graphNode.id)
        if (manager.graphNode.id == dataGraph.result.id) {
            log.debug("Result node has called onDataComputed so execution is complete.")
            executionResult.complete(data)
        } else {
            log.debug("Notifying subscribers of {}", manager.graphNode.id)
            notifySubscribers(manager, data, manager.graphNode.subscribers)
            if (execType == ExecutionType.TRAIN_TRANSFORM) {
                log.debug("Notifying train only subscribers of {}", manager.graphNode.id)
                notifySubscribers(manager, data, manager.graphNode.trainOnlySubscribers)
            } else {
                check(execType == ExecutionType.TRANSFORM)
            }
        }
    }

    fun onNodeFailed(manager: NodeExecutionManager, error: Throwable) {
        log.error("Execution of node {} failed:", manager.graphNode.id, error)
        executionResult.completeExceptionally(error)
    }

    private fun notifySubscribers(
            producingManager: NodeExecutionManager, data: DataSet, subscribers: List<GraphNode>) {
        for (sub in subscribers) {
            log.debug("Notifying {} that data is available", sub.id)
            // Find the index of this source in the subscribers list of sources
            val sourceIdx = sub.sources.indexOfFirst { it.id == producingManager.graphNode.id }
            check(sourceIdx >= 0) {
                "Subscriber node ${sub.id} does not list ${producingManager.graphNode.id} as a source."
            }
            log.debug("Output of node {} is input {} for node {}",
                    producingManager.graphNode.id, sourceIdx, sub.id)
            executionManagers[sub.id].onDataAvailable(sourceIdx, data)
        }

    }

    fun onReadyToRun(manager: NodeExecutionManager) {
        log.debug("Node {} reports that is is ready to run.", manager.graphNode.id)
        // By default ExecutorService will "swallow" an exceptions thrown and not log them or anything making debugging
        // very difficult. So we wrap the call to the manager with our own Runnable so we can catch exceptions and log.
        exec.submit {
            log.debug("Starting execution of node {}", manager.graphNode.id)
            try {
                manager.run().whenComplete { dataSet, throwable ->
                    if (throwable == null) {
                        onDataComputed(manager, dataSet)
                    } else {
                        log.error("Execution of node {} failed with:", manager.graphNode.id, throwable)
                        executionResult.completeExceptionally(throwable)
                    }
                }
            } catch (t: Throwable) {
                log.error("Execution of node {} failed:", manager.graphNode.id, t)
                executionResult.completeExceptionally(t)
            }
        }
    }
}

enum class ExecutionType {
    TRAIN_TRANSFORM, TRANSFORM
}

/**
 * Manages the execution of a single node in a graph for a single run on `execute`, `fit`, or `fitTransform`. See
 * the comments on [GraphExecution] for details.
 *
 * Note that when [onDataAvailable] is called the execution manager is in charge of maintaining a reference to the data
 * it will need in order to [run]. Once the run is complete that data is no longer needed by this node so it is a best
 * practice to remove the reference to the data so it can be GC'd.
 */
interface NodeExecutionManager {
    val graphNode: GraphNode
    /**
     * Called when data is available that this nodes requires.
     *
     * @param sourceIdx the index of the [GraphNode] that produced this data in the node's list of sources.
     * @param data the data that was produced.
     */
    fun onDataAvailable(sourceIdx: Int, data: DataSet)

    /**
     * When called the node should compute it's result. When the result has been computed the returned future should
     * complete with its value.
     *
     * Note that any exceptions thrown by this method will be handled by the framework.
     */
    fun run(): CompletableFuture<DataSet>
}

/**
 * Convenience base class for [NodeExecutionManager]s that are ready when a single [DataSet] is available. Subclasses
 * need only override [doRun] to perform the actual computation.
 */
abstract class SingleInputExecutionManager(protected val parent: GraphExecution) : NodeExecutionManager {
    @Volatile
    private var data: DataSet? = null

    companion object {
        private val log = LoggerFactory.getLogger(SingleInputExecutionManager::class.java)
    }

    override fun onDataAvailable(sourceIdx: Int, data: DataSet) {
        this.data = data
        parent.onReadyToRun(this)
    }

    final override fun run(): CompletableFuture<DataSet> {
        return doRun(data!!).whenComplete { dataSet, throwable ->
            // Get rid of our reference to the observaton so it can be GC'd if nothing else is using it.
            data = null
        }
    }

    abstract fun doRun(dataSet: DataSet): CompletableFuture<DataSet>

}
