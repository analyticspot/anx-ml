package com.analyticspot.ml.framework.description

/**
 * A value id allows the user to obtain data from a [DataSet], or [Observation] in a type safe way. Typically
 * [DataTransform] or [DataSet] generates it's own [ValueToken]s from [ValueId]. [ValueToken] is like [ValueId] except
 * that it also includes information about how to access the data efficiently. This allows each transform/data set to
 * create tokens that allow them to efficiently access their underlying data structure.
 *
 * By convention groups of related values are named with a prefix, a separator of `-` and a suffix. Thus, if you are
 * naming just a single feature/data item refrain from using the `-` character.
 */
open class ValueId<DataT>(val name: String, val clazz: Class<DataT>) : Comparable<ValueId<*>> {

    companion object {
        inline fun <reified T : Any> create(name: String) = ValueId<T>(name, T::class.java)
    }

    final override fun compareTo(other: ValueId<*>): Int {
        if (name == other.name) {
            check(clazz == other.clazz) {
                "Somehow there's two ValueToken instances named $name with different types: " +
                    "[$clazz] and [${other.clazz}]"
            }
            return 0
        } else {
            return name.compareTo(other.name)
        }
    }

    final override fun equals(other: Any?): Boolean {
        if (this === other) return true

        if (other is ValueId<*>) {
            return compareTo(other) == 0
        } else {
            return false
        }
    }

    final override fun hashCode(): Int {
        return name.hashCode()
    }

}
