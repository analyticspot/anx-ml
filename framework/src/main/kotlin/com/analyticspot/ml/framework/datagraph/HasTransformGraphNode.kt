package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.SingleDataTransform

/**
 * Base class for [GraphNode] instances which have a [SingleDataTransform] as a member.
 */
abstract class HasTransformGraphNode<T : SingleDataTransform>(builder: GraphNode.Builder) : GraphNode(builder) {
    abstract val transform: T
}
