package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

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
     * @param subId the value of the `subId` in the [Subscription] that produced this data.
     * @param data the data that was produced.
     */
    fun onDataAvailable(subId: Int, data: DataSet)

    /**
     * When called the node should compute it's result. When the result has been computed the returned future should
     * complete with its value.
     *
     * Note that any exceptions thrown by this method will be handled by the framework.
     */
    fun run(exec: ExecutorService): CompletableFuture<DataSet>
}
