package com.analyticspot.ml.framework.serialization

import com.fasterxml.jackson.core.JsonGenerator
import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind.DeserializationContext
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
        mapper = ObjectMapper().registerKotlinModule()
                .registerModule(kClassModule)
                .disable(JsonGenerator.Feature.AUTO_CLOSE_TARGET)
                .disable(JsonParser.Feature.AUTO_CLOSE_SOURCE)
    }

    object kClassModule : SimpleModule() {
        init {
            addSerializer(KClass::class.java, kClassSer)
            addDeserializer(KClass::class.java, kClassDeser)
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
                Boolean::class.qualifiedName to Boolean::class
        )

        override fun deserialize(parser: JsonParser, ctxt: DeserializationContext): KClass<*> {
            val classStr = parser.readValueAs(String::class.java)
            return kotlinTypesMap[classStr] ?: Class.forName(classStr).kotlin
        }
    }
}

