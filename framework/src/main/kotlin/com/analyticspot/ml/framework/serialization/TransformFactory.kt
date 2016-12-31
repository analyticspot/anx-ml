package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream

/**
 *
 */
interface TransformFactory<MetaDataT : FormatMetaData> {
    fun deserialize(formatMetaData: MetaDataT, sources: List<GraphNode>, input: InputStream): DataTransform
}
