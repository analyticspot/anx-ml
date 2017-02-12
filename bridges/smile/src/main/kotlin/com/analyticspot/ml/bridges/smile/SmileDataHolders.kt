package com.analyticspot.ml.bridges.smile

import smile.data.Attribute

/**
 * Simple wrapper for holding the arrays of double that define a data set in smile along with the attributes that
 * describe it. This is similar to smile's `AttributeDataSet` but with clearer semantics and it's much faster to
 * construct (`AttributeDataSet` must be constructed empty and then you must call `add` for each row which causes
 * array resizing, etc.)
 */
data class DataAndAttrs(val data: Array<DoubleArray>, val attributes: Array<Attribute>)

/**
 * Simple wrapper for holding an array of target data.
 */
data class CategoricalTarget(val target: IntArray, val stringToIntMapping: Map<String, Int>) {
    val intToStringMapping: Map<Int, String> by lazy {
        stringToIntMapping.asSequence().associate { it.value to it.key }
    }
}
