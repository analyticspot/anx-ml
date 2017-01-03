package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.datatransform.MultiTransform
import com.fasterxml.jackson.databind.InjectableValues
import org.slf4j.LoggerFactory
import java.io.InputStream
import java.io.OutputStream

/**
 * Our standard serialization format.
 */
class StandardJsonFormat : Format<StandardJsonFormat.MetaData> {
    override val metaDataClass: Class<MetaData>
        get() = MetaData::class.java

    companion object {
        private val log = LoggerFactory.getLogger(StandardJsonFormat::class.java)
    }

    override fun getMetaData(transform: DataTransform): MetaData {
        return MetaData(transform)
    }

    override fun serialize(transform: DataTransform, output: OutputStream) {
        JsonMapper.mapper.writeValue(output, transform)
    }

    override fun deserialize(metaData: MetaData, sources: List<GraphNode>, input: InputStream): DataTransform {
        val injectables = InjectableValues.Std()
        if (sources.size == 1) {
            injectables.addValue(GraphNode::class.java, sources[0])
        } else {
            injectables.addValue(MultiTransform.JSON_SOURCE_INJECTION_ID, sources)
            log.debug("Transform had {} sources so not automatically injecting a single source.", sources.size)
        }
        return JsonMapper.mapper.setInjectableValues(injectables).readValue(input, metaData.transformClass)
    }

    /**
     * The [FormatMetaData] for [StandardJsonFormat].
     */
    class MetaData(val transformClass: Class<out DataTransform>) : FormatMetaData {
        constructor(transform: DataTransform) : this(transform.javaClass)

        override val formatClass = StandardJsonFormat::class.java
    }
}
