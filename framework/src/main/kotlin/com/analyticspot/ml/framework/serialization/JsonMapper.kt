/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * The ANX ML library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.description.ColumnId
import com.fasterxml.jackson.core.JsonGenerator
import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind.DeserializationContext
import com.fasterxml.jackson.databind.KeyDeserializer
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializerProvider
import com.fasterxml.jackson.databind.deser.std.StdDeserializer
import com.fasterxml.jackson.databind.module.SimpleModule
import com.fasterxml.jackson.databind.ser.std.StdSerializer
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import kotlin.reflect.KClass

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
        mapper = ObjectMapper()
                .registerKotlinModule()
                .registerModule(kClassModule)
                .disable(JsonGenerator.Feature.AUTO_CLOSE_TARGET)
                .disable(JsonParser.Feature.AUTO_CLOSE_SOURCE)
    }

    object kClassModule : SimpleModule() {
        init {
            addSerializer(KClass::class.java, kClassSer)
            addDeserializer(KClass::class.java, kClassDeser)
            addKeyDeserializer(ColumnId::class.java, ColumnIdKeyDeserializer())
        }
    }

    // Custom serializer for KClass
    object kClassSer : StdSerializer<KClass<*>>(KClass::class.java) {
        override fun serialize(value: KClass<*>, gen: JsonGenerator, provider: SerializerProvider?) {
            gen.writeObject(value.qualifiedName)
        }
    }

    object kClassDeser : StdDeserializer<KClass<*>>(KClass::class.java) {
        // As per https://youtrack.jetbrains.com/issue/KT-10440 there isn't a reliable way to get back a KClass for a
        // Kotlin-specific type like kotlin.Int. So we use this ugly hack until that's fixed.
        val kotlinTypesMap = mapOf(
                Double::class.qualifiedName to Double::class,
                Float::class.qualifiedName to Float::class,
                Long::class.qualifiedName to Long::class,
                Int::class.qualifiedName to Int::class,
                Short::class.qualifiedName to Short::class,
                Byte::class.qualifiedName to Byte::class,
                String::class.qualifiedName to String::class,
                Boolean::class.qualifiedName to Boolean::class,
                List::class.qualifiedName to List::class,
                MutableList::class.qualifiedName to MutableList::class,
                Map::class.qualifiedName to Map::class,
                MutableMap::class.qualifiedName to MutableMap::class
        )

        override fun deserialize(parser: JsonParser, ctxt: DeserializationContext): KClass<*> {
            val classStr = parser.readValueAs(String::class.java)
            return kotlinTypesMap[classStr] ?: Class.forName(classStr).kotlin
        }
    }

    class ColumnIdKeyDeserializer : KeyDeserializer() {
        override fun deserializeKey(key: String, ctxt: DeserializationContext): ColumnId<*> {
            // Seems odd to use an ObjectMapper in the middle of deserializing stuff but I've seen this in several
            // KeyDeserializer examples and other options don't seem to work. Waiting on a response to
            // http://stackoverflow.com/questions/42497513/custom-keydeserializer
            return JsonMapper.mapper.readValue(key, ColumnId::class.java)
        }
    }
}

