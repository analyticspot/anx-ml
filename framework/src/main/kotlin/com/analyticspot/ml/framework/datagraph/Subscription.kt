package com.analyticspot.ml.framework.datagraph

/**
 * A [Subscription] indicates that a [GraphNode] requires the output of another [GraphNode]. Each [Subscription]
 * indicates the consuming node and an id. The source node then holds a reference to the subscription so that the
 * [GraphExecution] can notify subscribers when the data they require has been computed. The `subId` for a
 * [Subscription] is just an integer and is meaningful only to the subscriber. For example, a subscriber that has
 * subscribed to several data sets might assign each an index and use that as the `subId` so that when
 * [NodeExecutionManager] gets notified that some data is available it knows **which** data is now available.
 */
data class Subscription(val subscriber: GraphNode, val subId: Int)
