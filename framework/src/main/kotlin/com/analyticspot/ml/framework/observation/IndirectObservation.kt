package com.analyticspot.ml.framework.observation

import com.analyticspot.ml.framework.description.IndirectValueToken
import com.analyticspot.ml.framework.description.ValueToken

/**
 * An [Observation] that is just a combination of other [Observation] instances. Used for things like fast, efficient
 * merges of [DataSet] instances. These use [IndirectValueToken] to index into source observation and then uses that
 * [Observation]'s own token to retrieve the value.
 */
class IndirectObservation(private val sources: List<Observation>) : Observation {
    override val size: Int by lazy {
        sources.map { it.size }.sum()
    }

    override fun <T> value(token: ValueToken<T>): T {
        if (token is IndirectValueToken<T>) {
            check(token.obsIndex <= sources.size)
            return sources[token.obsIndex].value(token.obsToken)
        } else {
            throw IllegalStateException("Invalid token type. Require IndirectValueToken but found " + token.clazz.name)
        }
    }
}
