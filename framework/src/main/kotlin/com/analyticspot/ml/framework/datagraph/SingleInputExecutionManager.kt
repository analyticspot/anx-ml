package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Convenience base class for [NodeExecutionManager]s that are ready when a single [DataSet] is available. Subclasses
 * need only override [doRun] to perform the actual computation.
 */
abstract class SingleInputExecutionManager(protected val parent: GraphExecutionProtocol) : NodeExecutionManager {
    @Volatile
    private var data: DataSet? = null

    override fun onDataAvailable(subId: Int, data: DataSet) {
        this.data = data
        parent.onReadyToRun(this)
    }

    final override fun run(exec: ExecutorService): CompletableFuture<DataSet> {
        return doRun(data!!, exec).whenComplete { dataSet, throwable ->
            // Get rid of our reference to the observation so it can be GC'd if nothing else is using it.
            data = null
        }
    }

    abstract fun doRun(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet>
}
