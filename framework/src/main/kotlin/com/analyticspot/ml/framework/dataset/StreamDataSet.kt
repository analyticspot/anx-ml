package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.observation.Observation
import java.util.stream.Stream

/**
 * A [DataSet] constructed from a `Stream<Observation>`.
 */
class StreamDataSet(private val dataStream: Stream<Observation>) : DataSet {
    override fun iterator(): Iterator<Observation> = dataStream.iterator()
}
