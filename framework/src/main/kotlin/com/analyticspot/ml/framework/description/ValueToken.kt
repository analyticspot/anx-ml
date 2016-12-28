package com.analyticspot.ml.framework.description

/**
 * A [ValueToken] is a [ValueId] plus additional, hidden, information that allows the [Observation] or [DataSet] to
 * quickly access the underlying values. For example, if the data is stored in an array the [ValueToken] might contain
 * the integer index into the array. Typically a [DataTrasfrom] knows how its outputs are stored so the execute is
 * responsible for generating [ValueToken]s given [ValueId]s.
 */
open class ValueToken<DataT>(private val valId: ValueId<DataT>) {
    val name: String
        get() = valId.name
    val clazz: Class<DataT>
        get() = valId.clazz
    val id: ValueId<DataT>
        get() = valId
}
