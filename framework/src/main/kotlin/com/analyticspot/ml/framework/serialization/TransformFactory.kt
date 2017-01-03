package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream

/**
 * A function that can produce a [DataTransform] from serialized data. Registered with [GraphSerDeser] to do injection.
 */
interface TransformFactory<MetaDataT : FormatMetaData> {
    /**
     * Reads the data from `input` and deserializes it into a [DataTransform]. The `sources` indicate where the data
     * consumed by the [DataTransform] is produced and can be used to convert [ValueId] data into [ValueToken]
     * instances. The `metaData` describes what is found in the `input` and may help with deserialization.
     */
    fun deserialize(metaData: MetaDataT, sources: List<GraphNode>, input: InputStream): DataTransform
}
