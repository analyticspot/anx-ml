package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.observation.ArrayObservation
import com.analyticspot.ml.framework.observation.Observation

/**
 * A [DataSet] constructed from an `Iterable]<Observation>`.
 */
class IterableDataSet(private val dataSrc: Iterable<Observation>) : DataSet {

    companion object {
        /**
         * Construct an [IterableDataSet] from an `Iterable` of `Collection`. Note that all columns must have the same
         * data type and all rows must have the same size though this method does not check that explicitly.
         */
        fun create(data: Iterable<Collection<Any>>): IterableDataSet {
            val arrayOfObs = data.asSequence().map { ArrayObservation(it) }.toList()
            return IterableDataSet(arrayOfObs)
        }

    }

    override fun iterator(): Iterator<Observation> = dataSrc.iterator()
}
