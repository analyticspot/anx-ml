package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.observation.IndirectObservation
import com.analyticspot.ml.framework.observation.Observation

/**
 * A collection of [IndirectObservation]. See that class and [IndirectValueToken] for details.
 */
class IndirectDataSet(private val sources: List<DataSet>) : DataSet {
    override fun iterator(): Iterator<Observation> {
        return IndirectIterator(sources.map { it.iterator() })
    }

    private class IndirectIterator(private val sources: List<Iterator<Observation>>) : Iterator<Observation> {
        override fun hasNext(): Boolean {
            if (sources[0].hasNext()) {
                check(sources.all { it.hasNext() })
                return true
            } else {
                check(sources.none { it.hasNext() })
                return false
            }
        }

        override fun next(): Observation {
            val obs = sources.map { it.next() }
            return IndirectObservation(obs)
        }
    }
}
