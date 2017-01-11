package com.analyticspot.ml.framework.description

/**
 * A single id that refers to a set of [ValueId] all with the same type. There's at least two use cases for this:
 *
 * 1. Some algorithms produce an unknown number of values. For example, a "bag of words" transform will produce one
 * `Int` value for each unique word in the training corpus: until it's been trained we can't know many words there will
 * be. Still, we often want to tell consuming [DataTransformer] which values to consume from a [DataSet] so we need a
 * way to refer to a group of tokens. This is done via a [ValueIdGroup]: it refers to an unknown number of [ValueId]
 * that all have the same type and whose names all have the same prefix. See also [ValueTokenGroup].
 * 2. Many transforms can work on multiple id's (e.g. replace missing values with the mean in the training data for all
 * `Double` type values). Instead of passing an arrays of tokens mixed with `ValueIdGroup` to everything we can simply
 * pass a `ValueIdGroup`.
 *
 * Note that for use case (1) the corresponding [ValueTokenGroup] will not know the number of features or the actual
 * tokens until upstream algorithms have trained.
 */
open class ValueIdGroup<T>(val name: String, val clazz: Class<T>) : Comparable<ValueIdGroup<T>> {
    init {
        check(!name.contains(ValueId.GROUP_SEPARATOR)) {
            "ValueIdGroup prefixes should not contain the character ${ValueId.GROUP_SEPARATOR}"
        }
    }

    companion object {
        inline fun <reified T : Any> create(prefix: String): ValueIdGroup<T> {
            return ValueIdGroup(prefix, T::class.java)
        }
    }

    override fun compareTo(other: ValueIdGroup<T>): Int {
        return name.compareTo(other.name)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other?.javaClass != javaClass) return false

        other as ValueIdGroup<*>

        return name == other.name
    }

    override fun hashCode(): Int {
        return name.hashCode()
    }
}

