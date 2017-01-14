package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.serialization.JsonMapper
import kotlin.reflect.KClass

/**
 * A value id allows the user to obtain data from a [DataSet] in a type safe way.
 *
 * By convention groups of related values are named with a prefix, a separator of `-` and a suffix. Thus, if you are
 * naming just a single feature/data item refrain from using the `-` character.
 */
open class ColumnId<DataT : Any>(val name: String, val clazz: KClass<DataT>) : Comparable<ColumnId<*>> {
    /**
     * Alternative constructor taking Java classes.
     */
    constructor(name: String, clazz: Class<DataT>) : this(name, clazz.kotlin)

    companion object {
        inline fun <reified T : Any> create(name: String) = ColumnId<T>(name, T::class)
        /**
         * The character used to separate the prefix and the tokens in a [ValueTokenGroup].
         */
        const val GROUP_SEPARATOR = "-"
    }

    final override fun compareTo(other: ColumnId<*>): Int {
        if (name == other.name) {
            check(clazz.javaObjectType == other.clazz.javaObjectType) {
                "Somehow there's two ColumnId instances named $name with different types: " +
                    "[$clazz] and [${other.clazz}]"
            }
            return 0
        } else {
            return name.compareTo(other.name)
        }
    }

    final override fun equals(other: Any?): Boolean {
        if (this === other) return true

        if (other is ColumnId<*>) {
            return compareTo(other) == 0
        } else {
            return false
        }
    }

    final override fun hashCode(): Int {
        return name.hashCode()
    }

    override fun toString(): String {
        return JsonMapper.mapper.writeValueAsString(this)
    }
}
