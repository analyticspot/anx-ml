package com.analyticspot.ml.framework.observation

import com.analyticspot.ml.framework.description.ValueToken

/**
 *
 */
interface Observation {
    val size: Int
    fun <T> value(token: ValueToken<T>): T
}
