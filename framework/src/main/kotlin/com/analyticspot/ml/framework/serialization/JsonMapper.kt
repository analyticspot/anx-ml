package com.analyticspot.ml.framework.serialization

import com.fasterxml.jackson.core.JsonGenerator
import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.ser.impl.SimpleFilterProvider
import com.fasterxml.jackson.module.kotlin.registerKotlinModule

/**
 * It is recommended to use a single `ObjectMapper` for the entire executable as Jackson caches all the reflection
 * stuff. Also, we may want to register modules with the mapper to enable it to work better with Kotlin, to know how
 * to serialize various types, etc. Thus we use the singeton instance created here everywhere.
 *
 * Note: Unlike the normal `ObjectMapper` this is set to **not** auto-close `OutputStream` when it's done writing an
 * object or to close when it's done reading. This is because (1) that's unintuitive behavior and (2) our serialization
 * format is a zip file which means we read/write several values from/to a single `ZipOutputStream`.
 */
object JsonMapper {
    val mapper: ObjectMapper
    init {
        val filterProvider = SimpleFilterProvider()
        mapper = ObjectMapper().registerKotlinModule().setFilterProvider(filterProvider)
                .disable(JsonGenerator.Feature.AUTO_CLOSE_TARGET)
                .disable(JsonParser.Feature.AUTO_CLOSE_SOURCE)
    }
}

