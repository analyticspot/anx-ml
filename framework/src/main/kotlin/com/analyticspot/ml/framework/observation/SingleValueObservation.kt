package com.analyticspot.ml.framework.observation

import com.analyticspot.ml.framework.description.ValueToken

/**
 * An [Observation] that contains only a single value of type `<DataT>`.
 */
class SingleValueObservation<DataT>(private val value: DataT, private val clazz: Class<DataT>) : Observation {
    override val size: Int = 1

    companion object {
        inline fun <reified T : Any> create(value: T): SingleValueObservation<T> =
                SingleValueObservation(value, T::class.java)
    }

    override fun <T> value(token: ValueToken<T>): T {
        check(token.clazz.isAssignableFrom(clazz))
        @Suppress("UNCHECKED_CAST")
        return value as T
    }
}
