package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.observation.Observation

/**
 * A [DataSet] that contains an array of [Observation].
 */
class ArrayDataSet(private val data: Array<out Observation>) : DataSet {
    override fun iterator(): Iterator<Observation> = data.iterator()

    override fun toArray(): Array<out Observation> = data
}
