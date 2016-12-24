package com.analyticspot.ml.framework.description

/**
 * A value token allows the user to obtain data from a [DataSet], [DataDescription], or [Observation] in a type safe
 * way. Typically [DataTransform] or [DataSet] generates it's own tokens. This allows each transform/data set to create
 * tokens that allow them to efficiently access their underlying data structure.
 *
 * By convention groups of related values are named with a prefix, a separator of `-` and a suffix. Thus, if you are
 * naming just a single feature/data item refrain from using the `-` character.
 */
interface ValueToken<DataT> : Comparable<ValueToken<*>> {
    val name: String
    val clazz: Class<DataT>
}
