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
    fun intercept(subIdToData: Map<Int, DataSet>, execType: ExecutionType, output: DataSet): CompletableFuture<DataSet>
}
