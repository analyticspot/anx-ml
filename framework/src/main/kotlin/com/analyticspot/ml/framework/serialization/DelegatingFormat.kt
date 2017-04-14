package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import java.io.InputStream
import java.io.OutputStream

/**
 * Format for [DataTransform] class that simply delegate to another [DataTransform]. There's a variety of reasons we
 * might want to do that:
 *
 * * Perhaps the training code requires several big libraries but using the trained model does not. So, to cut down
 *   on runtime dependencies the transform might serialize as a different class that knows only how to score, but not
 *   to train. Thus serialization would simply delegate to that subclass.
 * * Sometimes it's helpful to have the [DataSet] available in order to configure a transform (e.g. so you know exactly
 *   which columns are in the [DataSet]). In this case you can configure the transform in `trainTransform` and then
 *   simply delegate the `transform` method to the delegate.
 *
 * Classes that want to use this format must implement [DelegatingTransform].
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
