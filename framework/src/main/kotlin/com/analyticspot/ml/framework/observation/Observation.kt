package com.analyticspot.ml.framework.observation

import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.description.ValueTokenGroup

/**
 * An [Observation] represents a single data point from which to make a prediction or to learn. An [Observation] may
 * consist of multiple values. Many machine learning texts and programs organize data sets into rows and columns with
 * one row per observation and one column per feature. In this context an [Observation] is a row and a [value] is the
 * value in the column indicated by a [ValueToken] for that row.
 */
interface Observation {
    /**
     * Returns the number of values in the [Observation].
     */
    val size: Int

    /**
     * Returns the value for the given [ValueToken].
     */
    fun <T> value(token: ValueToken<T>): T

    /**
     * Returns all the values for the [ValueTokenGroup]. The values are returns in the same order as
     * returned by [ValueTokenGroup.tokens].
     */
    fun <T> values(tokenGroup: ValueTokenGroup<T>): List<T> {
        return tokenGroup.tokens().map { value(it) }
    }
}
