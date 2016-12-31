package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
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
            log.info("Transform had {} sources so not automatically injecting the source.", sources.size)
        }
        return JsonMapper.mapper.setInjectableValues(injectables).readValue(input, metaData.transformClass)
    }

    class MetaData(tranform: DataTransform) : FormatMetaData {
        val transformClass = tranform.javaClass
        override val formatClass = StandardJsonFormat::class.java
    }
}
