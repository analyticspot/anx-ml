package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream
import java.io.OutputStream

/**
 *
 */
interface Format<MetaDataT : FormatMetaData> {
    val metaDataClass: Class<MetaDataT>
    /**
     * Returns a JSON-serializable object that is the metadata for the format.
     */
    fun getMetaData(transform: DataTransform) : MetaDataT

    fun serialize(transform: DataTransform, output: OutputStream)

    fun deserialize(metaData: MetaDataT, sources: List<GraphNode>, input: InputStream): DataTransform
}
