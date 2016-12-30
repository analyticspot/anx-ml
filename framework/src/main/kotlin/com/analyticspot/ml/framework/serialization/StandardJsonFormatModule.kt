package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datatransform.DataTransform
import com.fasterxml.jackson.databind.InjectableValues
import java.io.InputStream
import java.io.OutputStream

/**
 *
 */
class StandardJsonFormatModule : FormatModule<StandardJsonData> {
    private val factories: MutableMap<String, TransformFactory<StandardJsonData>> = mutableMapOf()

    private object standardFactory : TransformFactory<StandardJsonData> {
        override fun deserialize(
                formatData: StandardJsonData, sources: List<GraphNode>, input: InputStream): DataTransform {
            val injectables = InjectableValues.Std()
            if (sources.size == 1) {
                injectables.addValue(GraphNode::class.java, sources[0])
            }
            return JsonMapper.mapper.setInjectableValues(injectables).readValue(input, formatData.transformClass)
        }
    }

    override fun registerFactory(tag: String, factory: TransformFactory<StandardJsonData>) {
        check(!factories.containsKey(tag)) {
            "A FormatFactory for tag $tag has already been registered."
        }
        factories[tag] = factory
    }

    override fun formatData(transform: DataTransform): FormatData {
        return StandardJsonData(transform.javaClass)
    }

    override fun serialize(transform: DataTransform, output: OutputStream) {
        JsonMapper.mapper.writeValue(output, transform)
    }

    override fun getFactory(tag: String?): TransformFactory<StandardJsonData> {
        if (tag != null && factories.containsKey(tag)) {
            return factories[tag] ?: throw IllegalStateException("Somehow found a null value.")
        } else {
            return standardFactory
        }
    }
}
