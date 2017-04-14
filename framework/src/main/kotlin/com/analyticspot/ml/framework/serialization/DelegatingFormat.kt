package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream
import java.io.OutputStream

/**
 * Created by oliver on 4/6/17.
 */
class DelegatingFormat : Format {

    override fun getMetaData(transform: DataTransform, serDeser: GraphSerDeser): FormatMetaData {
        if (transform is DelegatingTransform) {
            val delegateFormat = serDeser.formatClassToFormat[transform.delegate.formatClass]!!
            return delegateFormat.getMetaData(transform.delegate, serDeser)
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

    override fun deserialize(metaData: FormatMetaData, sources: List<GraphNode>,
            serDeser: GraphSerDeser, input: InputStream): DataTransform {
        throw IllegalStateException("DelegatingFormat should never be deserializing: the delegate should handle it.")
    }

}
