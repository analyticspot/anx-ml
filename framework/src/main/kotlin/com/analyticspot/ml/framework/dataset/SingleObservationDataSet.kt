package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.observation.Observation

/**
 * A [DataSet] that holds just a single [Observation].
 */
class SingleObservationDataSet(observation: Observation) : DataSet {
    private val data = arrayOf(observation)
    override fun iterator(): Iterator<Observation> {
        return data.iterator()
    }
}
