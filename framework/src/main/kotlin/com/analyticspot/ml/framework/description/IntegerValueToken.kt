package com.analyticspot.ml.framework.description

/**
 * A [ValueToken] with an integer `index` member. Often used as an index into an `Array` of values by
 * [ArrayObservation].
 */
class IntegerValueToken<DataT>(val index: Int, name: String, clazz: Class<DataT>) : ValueToken<DataT>(name, clazz) {
}
