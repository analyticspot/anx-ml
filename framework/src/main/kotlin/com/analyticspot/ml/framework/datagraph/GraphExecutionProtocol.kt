package com.analyticspot.ml.framework.datagraph

/**
 * See the protocol description on [GraphExecution].
 */
interface GraphExecutionProtocol {
    fun onReadyToRun(manager: NodeExecutionManager)
}
