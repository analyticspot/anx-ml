package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.StreamDataSet
import com.analyticspot.ml.framework.observation.Observation
import kotlinx.support.jdk8.collections.spliterator
import java.util.stream.StreamSupport

/**
 * An abstract base class for [DataTransform] that allows users to override a method that takes a single [Observation]
 * and returns a single [Observation] instead of having to worry about entire [DataSet]s.
 */
abstract class StreamingDataTransform : DataTransform {
    final override fun transform(dataSet: DataSet): DataSet {
        val obsStream = StreamSupport.stream(dataSet.spliterator(), false).map { transform(it) }
        return StreamDataSet(obsStream)
    }

    abstract fun transform(observation: Observation): Observation
}
