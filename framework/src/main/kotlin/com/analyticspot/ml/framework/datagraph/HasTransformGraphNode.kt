package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.description.TransformDescription

/**
 * Base class for [GraphNode] instances which have a [SingleDataTransform] as a member.
 */
abstract class HasTransformGraphNode<out T : DataTransform>(builder: GraphNode.Builder) : GraphNode(builder) {
    abstract val transform: T
    override val transformDescription: TransformDescription
        get() = transform.description
}
