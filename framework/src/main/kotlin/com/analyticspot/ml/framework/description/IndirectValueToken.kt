package com.analyticspot.ml.framework.description

/**
 * A [ValueToken] that gets its value from another, underlying [Observation]. This allows us to do things like a
 * [MergeTransform] very efficiently; intead of copying the data we simply point to the original source data.
 *
 * See [IndirectObservation] for details.
 *
 * @param obsIndex the index of the underlying [Observation] from which the data comes.
 * @param obsToken the token that can be used to retrieve the data from the source [Observation].
 */
class IndirectValueToken<DataT>(val obsIndex: Int, val obsToken: ValueToken<DataT>)
    : ValueToken<DataT>(obsToken.id) {
}
