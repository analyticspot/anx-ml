package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import java.util.concurrent.CompletableFuture

/**
 * [OutputInterceptor]s can wrap a node in the [DataGraph]. They are called when the wrapped node completes execution
 * and are passed the inputs to that node and the output of the node. They can then modify the [DataSet] computed by
 * the wrapped node. The rest of the graph will see only the value returned by the interceptor; they will not see the
 * data computed by the wrapped node.
 */
interface OutputInterceptor {
    /**
     * Called when the execution of the wrapped node is complete.
     *
     * @param subIdToData contains the data that was the input to the wrapped node. It is a map from the subscription
     *     id to the data received by that subscription.
     * @param execType the type of execution (training or transform) being run
     * @param output the data computed by the wrapped transform
     *
     * @return a new [DataSet]. All graph nodes subscribing to the wrapped node will see the data this returns instead
     *     of the data returned by the wrapped node.
     */
    fun intercept(subIdToData: Map<Int, DataSet>, execType: ExecutionType, output: DataSet): CompletableFuture<DataSet>
}
