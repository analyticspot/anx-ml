package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import org.slf4j.LoggerFactory
import java.io.InputStream
import java.io.OutputStream

/**
 * Serializes and deserializes entire [DataGraph] instances. See `SERIALIZATION.README.md` for details.
 */
class GraphSerDeser {
    private val labelToDeserializer = mutableMapOf<String, TransformFactory<*>>()
    private val formatMap = mutableMapOf<Class<out Format<*>>, Format<*>>(
            StandardJsonFormat::class.java to StandardJsonFormat()
    )

    companion object {
        private val log = LoggerFactory.getLogger(GraphSerDeser::class.java)
    }

    fun registerFactoryForLabel(label: String, factory: TransformFactory<*>) {
        check(!labelToDeserializer.containsKey(label)) {
            "There is already factory registered for label $label"
        }
        labelToDeserializer[label] = factory
    }

    fun registerFormat(format: Format<*>) {
        val clazz = format.javaClass
        check(!formatMap.containsKey(clazz)) {
            "There is already a format registered with type $clazz"
        }
        formatMap[clazz] = format
    }

    // Visible for testing
    internal fun serializeTransform(transform: DataTransform, output: OutputStream) {
        val format = formatMap[transform.formatClass] ?:
                throw IllegalStateException("${transform.formatClass} is not registered")
        format.serialize(transform, output)
    }

    // Visible for testing
    internal fun <MetaDataT : FormatMetaData> deserializeTransform(label: String?, metaData: MetaDataT,
            sources: List<GraphNode>, input: InputStream): DataTransform {
        if (label != null && labelToDeserializer.containsKey(label)) {
            log.info("Using custom TransformFactory for label {}", label)
            val factory = labelToDeserializer[label] ?: throw IllegalStateException("Should be impossible")
            @Suppress("UNCHECKED_CAST")
            val typedFactory = factory as TransformFactory<MetaDataT>
            return typedFactory.deserialize(metaData, sources, input)
        } else {
            val format = formatMap[metaData.formatClass] ?:
                    throw IllegalStateException("Unknown format ${metaData.formatClass}")
            if (format.metaDataClass == metaData.javaClass) {
                @Suppress("UNCHECKED_CAST")
                val typedFormat = format as Format<MetaDataT>
                return typedFormat.deserialize(metaData, sources, input)
            } else {
                throw IllegalStateException("Format ${format.javaClass} expects metadata of type " +
                        "${format.metaDataClass} but found meta data of type ${metaData.javaClass}")
            }
        }
    }
}
