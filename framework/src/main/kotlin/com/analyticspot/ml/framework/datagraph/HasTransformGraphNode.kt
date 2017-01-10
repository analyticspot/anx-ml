package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.DataTransform

/**
 * Base class for [GraphNode] instances which have a [SingleDataTransform] as a member.
 */
abstract class HasTransformGraphNode<T : DataTransform>(builder: GraphNode.Builder) : GraphNode(builder) {
    abstract val transform: T
}
