package com.analyticspot.ml.bridges.smile

import smile.data.Attribute

/**
 * Simple wrapper for holding the arrays of double that define a data set in smile along with the attributes that
 * describe it. This is similar to smile's `AttributeDataSet` but with clearer semantics and it's much faster to
 * construct (`AttributeDataSet` must be constructed empty and then you must call `add` for each row which causes
 * array resizing, etc.)
 */
data class DataAndAttrs(val data: Array<DoubleArray>, val attributes: Array<Attribute>)
