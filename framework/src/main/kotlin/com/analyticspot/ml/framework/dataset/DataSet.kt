package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.observation.Observation

/**
 * An `Iterable<Observation>` representing a set of data.
 */
interface DataSet : Iterable<Observation> {
    fun toArray(): Array<out Observation> = this.toArray()
}

