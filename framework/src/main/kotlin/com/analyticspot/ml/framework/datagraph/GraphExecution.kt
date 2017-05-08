/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * The ANX ML library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Manages the execution of a single [DataSet] through the graph. See the PUSH.OR.PULL.README.md file in this
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
 * * When data is available the [NodeExecutionManager.onDataAvailable] method will be called for all nodes that have
 *   subscribed to that data.
 * * When a [NodeExecutionManager] has all the data it needs to run that manager will call [onReadyToRun] on this
 *   instance. This will then schedule the node to be run (often on another thread).
 * * Once scheduled the [NodeExecutionManager.run] method wil be called. The [NodeExecutionManager] will then call its
 *   underlying [DataTransform] to generate the data.
 * * Once the run is complete and the data is available the future returned by [NodeExecutionManager.run] will complete
 *   causing any subscribing nodes to be notified.
 *
 *   @param dataGraph the graph to be executed.
 *   @param execType the type of execution (training or transform).
 *   @param exec the `ExecutorService` to be used for scheduling the node executions.
 *   @param interceptors a map from node label to [OutputInterceptor]. Each node in the map will have its output
 *       intercepted by the given interceptor before it is made available to any other nodes.
 */
class GraphExecution (
        private val dataGraph: DataGraph, private val execType: ExecutionType,
        private val exec: ExecutorService,
        private val interceptors: Map<String, OutputInterceptor>? = null) : GraphExecutionProtocol {
    // Can be null as there may be missing nodes if we deserialize a graph that had train-only nodes.
    private val executionManagers: Array<NodeExecutionManager?>
    private val executionResult: CompletableFuture<DataSet> = CompletableFuture()

    init {
        if (dataGraph.source.label != null && interceptors != null) {
            require(!interceptors.containsKey(dataGraph.source.label!!)) {
                "Interceptors are not supported for the DataGraph's source. You can simply modify the data directly " +
                        "if necessary."
            }
        }

        val graphNodes = dataGraph.allNodes
        val matchedInterceptors = mutableSetOf<String>()
        executionManagers = Array<NodeExecutionManager?>(graphNodes.size) { idx ->
            val node = graphNodes[idx]
            if (interceptors != null && node?.label in interceptors) {
                log.info("Adding OutputInterceptor for node with label {}", node!!.label)
                matchedInterceptors.add(node.label!!)
                val factory = { proto: GraphExecutionProtocol, et: ExecutionType ->
                    graphNodes[idx]!!.getExecutionManager(proto, et)
                }

                OutputInterceptorExecManager(factory, this, interceptors[node.label!!]!!, execType)
            } else {
                graphNodes[idx]?.getExecutionManager(this, execType)
            }
        }

        check(interceptors == null || matchedInterceptors.size == interceptors.size) {
            "Some of the specified interceptors did not match any nodes. The following did not match: " +
                    interceptors!!.keys.subtract(matchedInterceptors).toString()
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(GraphExecution::class.java)
    }

    fun execute(data: DataSet): CompletableFuture<DataSet> {
        log.debug("Starting execution of DataGraph")
        onDataComputed(executionManagers[dataGraph.source.id]!!, data)
        return executionResult
    }

    fun onDataComputed(manager: NodeExecutionManager, data: DataSet) {
        log.debug("Node {} has finished computing its data.", manager.graphNode.id)
        if (manager.graphNode.id == dataGraph.result.id) {
            log.debug("Result node has called onDataComputed so execution is complete.")
            executionResult.complete(data)
        } else {
            log.debug("Notifying subscribers of {}", manager.graphNode.id)
            notifySubscribers(data, manager.graphNode.subscribers)
            if (execType == ExecutionType.TRAIN_TRANSFORM) {
                log.debug("Notifying train only subscribers of {}", manager.graphNode.id)
                notifySubscribers(data, manager.graphNode.trainOnlySubscribers)
            } else {
                check(execType == ExecutionType.TRANSFORM)
            }
        }
    }

    fun onNodeFailed(manager: NodeExecutionManager, error: Throwable) {
        log.error("Execution of node {} failed:", manager.graphNode.id, error)
        executionResult.completeExceptionally(error)
    }

    private fun notifySubscribers(data: DataSet, subscribers: List<Subscription>) {
        for (sub in subscribers) {
            log.debug("Notifying {} that data is available", sub.subscriber.id)
            executionManagers[sub.subscriber.id]!!.onDataAvailable(sub.subId, data)
        }

    }

    override fun onReadyToRun(manager: NodeExecutionManager) {
        log.debug("Node {} reports that is is ready to run.", manager.graphNode.id)
        // By default ExecutorService will "swallow" an exceptions thrown and not log them or anything making debugging
        // very difficult. So we wrap the call to the manager with our own Runnable so we can catch exceptions and log.
        exec.submit {
            log.debug("Starting execution of node {}", manager.graphNode.id)
            try {
                manager.run(exec).whenComplete { dataSet, throwable ->
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

