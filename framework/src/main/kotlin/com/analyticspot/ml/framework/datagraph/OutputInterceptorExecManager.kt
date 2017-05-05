package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Wraps another graph node so we can intercept and modify the output. For that to work we can't let the wrapped node
 * communicate directly with the GraphExecution; if it did it'd call [GraphExecution.onReadyToRun] causing the wrapped
 * node to be run by the [GraphExecution] directly and we then wouldn't have the opportunity to intercept and modify
 * the output. Thus, this implements [GraphExecutionProtocol] and it constructs the wrapped manager via
 * `wrappedManagerFactor` passing itself as the [GraphExecutionProtocol].
 */
class OutputInterceptorExecManager(
        wrappedManagerFactory: (GraphExecutionProtocol, ExecutionType) -> NodeExecutionManager,
        private val parent: GraphExecutionProtocol,
        private val interceptor: OutputInterceptor,
        private val execType: ExecutionType) : NodeExecutionManager, GraphExecutionProtocol {

    private val wrappedManager: NodeExecutionManager = wrappedManagerFactory.invoke(this, execType)

    companion object {
        private val log = LoggerFactory.getLogger(OutputInterceptorExecManager::class.java)
    }

    override val graphNode: GraphNode
        get() = wrappedManager.graphNode

    private var subIdToData = mutableMapOf<Int, DataSet>()

    override fun onDataAvailable(subId: Int, data: DataSet) {
        subIdToData[subId] = data
        wrappedManager.onDataAvailable(subId, data)
    }

    override fun run(exec: ExecutorService): CompletableFuture<DataSet> {
        return wrappedManager.run(exec).thenCompose {
            interceptor.intercept(subIdToData, execType, it).whenComplete { dataSet, exception ->
                // Drop references to the data so it can be GC'd
                subIdToData.clear()
                if (exception != null) {
                    log.error("Execution of NodeInterceptor failed with:", exception)
                }
            }
        }
    }

    override fun onReadyToRun(manager: NodeExecutionManager) {
        parent.onReadyToRun(this)
    }
}
