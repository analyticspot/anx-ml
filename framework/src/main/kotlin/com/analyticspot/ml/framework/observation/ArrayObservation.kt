package com.analyticspot.ml.framework.observation

import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueToken

/**
 * An [Observation] backed by a simple Array.
 */
class ArrayObservation : Observation {
    val data: Array<out Any>

    constructor(data: Array<out Any>) {
        this.data = data
    }

    constructor(data: Collection<out Any>) : this(data.toTypedArray())

    companion object {
        fun create(vararg data: Any): ArrayObservation {
            return ArrayObservation(data)
        }
    }

    override val size: Int
        get() = data.size

    override fun <T> value(token: ValueToken<T>): T {
        if (token is IndexValueToken<T>) {
            val v = data[token.index]
            return token.clazz.cast(v)
        } else {
            throw IllegalArgumentException("ArrayObservation expects only IntegerValueToken to be passed to value()")
        }
    }
}
