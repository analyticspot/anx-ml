package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream
import java.io.OutputStream

/**
 * Created by oliver on 4/6/17.
 */
class DelegatingFormat : Format<DelegatingFormat.DelegatingMetaData> {
    override val metaDataClass: Class<DelegatingMetaData> = DelegatingMetaData::class.java

    override fun getMetaData(transform: DataTransform): DelegatingMetaData {
        if (transform is DelegatingTransform) {
            return DelegatingMetaData(transform.delegate.formatClass)
        } else {
            throw IllegalArgumentException("Any class that declares DelegatingFormat must implement the " +
                    "DelegatingTransform interface")

        }
    }


    override fun serialize(transform: DataTransform, serDeser: GraphSerDeser, output: OutputStream) {
        if (transform is DelegatingTransform) {
            serDeser.serializeTransform(transform.delegate, output)
        } else {
            throw IllegalArgumentException("Any class that declares DelegatingFormat must implement the " +
                    "DelegatingTransform interface")
        }
    }

    override fun deserialize(metaData: DelegatingMetaData, sources: List<GraphNode>,
            serDeser: GraphSerDeser, input: InputStream): DataTransform {
        return serDeser.deserializeTransform(null, metaData.wrappedMetaData, sources, input)
    }

    class DelegatingMetaData(val wrappedMetaData: FormatMetaData) : FormatMetaData {
        override val formatClass = DelegatingFormat::class.java
    }
}
