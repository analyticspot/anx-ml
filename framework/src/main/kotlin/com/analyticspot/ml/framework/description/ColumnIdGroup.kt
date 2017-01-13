package com.analyticspot.ml.framework.description

/**
 * A single id that refers to a set of [ColumnId] all with the same type. This is used by some algorithms which produce
 * an unknown number of values. For example, a "bag of words" transform will produce one `Int` value for each unique
 * word in the training corpus: until it's been trained we can't know many words there will be. Still, we often want to
 * tell consuming [DataTransformer] which values to consume from a [DataSet] so we need a way to refer to a group of
 * columns. This is done via a [ColumnIdGroup]: it refers to an unknown number of [ColumnId] that all have the same type
 * and whose names all have the same prefix.
 */
open class ColumnIdGroup<T>(val prefix: String, val clazz: Class<T>) : Comparable<ColumnIdGroup<T>> {
    init {
        check(!prefix.contains(ColumnId.GROUP_SEPARATOR)) {
            "ValueIdGroup prefixes should not contain the character ${ColumnId.GROUP_SEPARATOR}"
        }
    }

    companion object {
        inline fun <reified T : Any> create(prefix: String): ColumnIdGroup<T> {
            return ColumnIdGroup(prefix, T::class.java)
        }
    }

    override fun compareTo(other: ColumnIdGroup<T>): Int {
        return prefix.compareTo(other.prefix)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other?.javaClass != javaClass) return false

        other as ColumnIdGroup<*>

        return prefix == other.prefix
    }

    override fun hashCode(): Int {
        return prefix.hashCode()
    }
}

