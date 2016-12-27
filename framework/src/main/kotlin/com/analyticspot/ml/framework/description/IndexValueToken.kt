package com.analyticspot.ml.framework.description

/**
 * A [ValueToken] with an integer `index` member. Often used as an index into an `Array` of values by
 * [ArrayObservation].
 */
class IndexValueToken<DataT>(val index: Int, valId: ValueId<DataT>) : ValueToken<DataT>(valId) {
    companion object {
        fun <T> create(index: Int, valId: ValueId<T>) = IndexValueToken(index, valId)
    }
}
