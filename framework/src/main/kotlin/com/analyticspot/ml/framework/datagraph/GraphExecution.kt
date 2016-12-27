package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.observation.Observation
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
 * * Once the run is complete and the data is available (e.g. after any `CompletableFuture` instances have completed)
 *   the [NodeExecutionManager] will call [onDataComputed] on this instance which will cause the entire process to
 *   repeat.
 */
class GraphExecution (private val dataGraph: DataGraph, private val exec: ExecutorService) {
    private val executionManagers: Array<NodeExecutionManager>
    private val executionResult: CompletableFuture<Observation> = CompletableFuture()

    init {
        val graphNodes = dataGraph.allNodes
        executionManagers = Array<NodeExecutionManager>(graphNodes.size) { idx ->
            graphNodes[idx].getExecutionManager(this)
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(GraphExecution::class.java)
    }

    fun transform(observation: Observation): CompletableFuture<Observation> {
        log.debug("Starting execution of DataGraph")
        for (sub in dataGraph.source.subscribers) {
            log.debug("Notifying node {} that the source is available", sub.id)
            executionManagers[sub.id].onDataAvailable(observation)
        }
        return executionResult
    }

    fun onDataComputed(manager: NodeExecutionManager, observation: Observation) {
        log.debug("Node {} has finished computing its Observation.", manager.graphNode.id)
        if (manager.graphNode.id == dataGraph.result.id) {
            log.debug("Result node has called onDataComputed so execution is complete.")
            executionResult.complete(observation)
        } else {
            for (sub in manager.graphNode.subscribers) {
                log.debug("Notifying {} that the data from {} is available", sub.id, manager.graphNode.id)
                executionManagers[sub.id].onDataAvailable(observation)
            }
        }
    }

    fun onReadyToRun(manager: NodeExecutionManager) {
        exec.submit(manager)
    }
}

/**
 * Manages the execution of a single node in a graph for a single run on `transform`, `fit`, or `fitTransform`. See
 * the comments on [GraphExecution] for details.
 *
 * Note that when [onDataAvailable] is called the execution manager is in charge of maintaining a reference to the data
 * it will need in order to [run]. Once the run is complete that data is no longer needed by this node so it is a best
 * practice to remove the reference to the data so it can be GC'd.
 */
interface NodeExecutionManager : Runnable {
    val graphNode: GraphNode
    fun onDataAvailable(observation: Observation)
}
