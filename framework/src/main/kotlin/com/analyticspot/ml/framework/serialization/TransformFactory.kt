package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream

/**
 *
 */
interface TransformFactory<T : FormatData> {
    fun deserialize(formatData: T, sources: List<GraphNode>, input: InputStream): DataTransform
}
