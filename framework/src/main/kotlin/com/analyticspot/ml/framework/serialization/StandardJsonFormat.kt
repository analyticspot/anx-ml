/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * Foobar is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

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
