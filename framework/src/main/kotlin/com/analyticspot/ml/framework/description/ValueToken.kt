package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.serialization.JsonMapper
import com.fasterxml.jackson.annotation.JacksonInject
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonFilter
import com.fasterxml.jackson.annotation.JsonProperty

/**
 * A [ValueToken] is a [ValueId] plus additional, hidden, information that allows the [Observation] or [DataSet] to
 * quickly access the underlying values. For example, if the data is stored in an array the [ValueToken] might contain
 * the integer index into the array. Typically a [DataTrasfrom] knows how its outputs are stored so the execute is
 * responsible for generating [ValueToken]s given [ValueId]s.
 */
@JsonFilter(JsonMapper.VALUE_TOKEN_FILTER_ID)
open class ValueToken<DataT>(private val valId: ValueId<DataT>) {
    val name: String
        get() = valId.name
    val clazz: Class<DataT>
        get() = valId.clazz
    val id: ValueId<DataT>
        get() = valId

    companion object {
        /**
         * Handy way to create a [ValueToken] from a [GraphNode] and a [ValueId]. This is particularly handy with
         * Jackson Json serialization as it allows us to inject the [GraphNode] that is the source for a transform
         * and have the serialized version of the [ValueToken] resolve to a token that's correct for the actual source.
         * See [JsonMapper] for more info.
         */
        @JsonCreator
        @JvmStatic
        fun <T> tokenFromSource(
                @JacksonInject source: GraphNode,
                @JsonProperty("id") valId: ValueId<T>): ValueToken<T> {
            return source.token(valId)
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other is ValueToken<*>) {
            check(other.clazz == clazz) {
                "Two value tokens named $name but with different types: $clazz and ${other.clazz}"
            }
            return name == other.name
        } else {
            return false
        }
    }

    override fun hashCode(): Int {
        return name.hashCode()
    }
}
