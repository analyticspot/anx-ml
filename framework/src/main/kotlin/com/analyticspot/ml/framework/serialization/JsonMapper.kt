package com.analyticspot.ml.framework.serialization

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.module.SimpleModule
import com.fasterxml.jackson.databind.ser.impl.SimpleBeanPropertyFilter
import com.fasterxml.jackson.databind.ser.impl.SimpleFilterProvider
import com.fasterxml.jackson.module.kotlin.registerKotlinModule

/**
 * It is recommended to use a single `ObjectMapper` for the entire executable as Jackson caches all the reflection
 * stuff. Also, we may want to register modules with the mapper to enable it to work better with Kotlin, to know how
 * to serialize various types, etc. Thus we use the singeton instance created here everywhere.
 */
object JsonMapper {
    val mapper: ObjectMapper
    // Used with @JsonFilter annotation on ValueToken so that we can be sure to filter only the ValueId part of it.
    const val VALUE_TOKEN_FILTER_ID = "VALUE_TOKEN_FILTER"

    init {
        val tokenFilter = SimpleBeanPropertyFilter.filterOutAllExcept("id")
        val filterProvider = SimpleFilterProvider().addFilter(VALUE_TOKEN_FILTER_ID, tokenFilter)
        mapper = ObjectMapper().registerKotlinModule().registerModule(AnxMlSerializationModule())
                .setFilterProvider(filterProvider)

    }
}

class AnxMlSerializationModule : SimpleModule(NAME) {
    companion object {
        private val NAME = AnxMlSerializationModule::class.java.name
    }

    override fun setupModule(context: SetupContext) {

    }
}

// class ValueTokenDeserializer : JsonDeserializer<ValueToken<*>>() {
//     override fun deserialize(parser: JsonParser, ctxt: DeserializationContext): ValueToken<*> {
//         val valueId = parser.readValueAs(ValueId::class.java)
//         val source = ctxt.findInjectableValue()
//     }
//
// }

