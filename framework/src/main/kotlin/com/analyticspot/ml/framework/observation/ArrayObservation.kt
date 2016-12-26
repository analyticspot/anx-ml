package com.analyticspot.ml.framework.observation

import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken

/**
 * An [Observation] backed by a simple Array.
 */
class ArrayObservation : Observation {
    val data: Array<Any>

    constructor(data: Array<Any>) {
        this.data = data
    }

    override val size: Int
        get() = data.size

    override fun <T> value(token: ValueToken<T>): T {
        if (token is IndexValueToken<T>) {
            val v = data[token.index]
            check(token.clazz.isAssignableFrom(v.javaClass)) {
                "value was called with T = ${token.clazz} but the found value was of type ${v.javaClass}"
            }
            @Suppress("UNCHECKED_CAST")
            return v as T
        } else {
            throw IllegalArgumentException("ArrayObservation expects only IntegerValueToken to be passed to value()")
        }
    }
}
