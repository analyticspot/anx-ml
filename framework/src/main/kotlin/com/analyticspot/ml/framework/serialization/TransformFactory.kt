package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream

/**
 *
 */
interface TransformFactory {
    fun createTransform(transformData: InputStream, sources: List<GraphNode>): DataTransform
}
