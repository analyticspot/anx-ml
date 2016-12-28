package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.observation.Observation
import kotlinx.support.jdk8.collections.spliterator
import kotlinx.support.jdk8.streams.toList
import java.util.stream.StreamSupport

/**
 * An abstract base class for [DataTransform] that allows users to override a method that takes a single [Observation]
 * and returns a single [Observation] instead of having to worry about entire [DataSet]s.
 */
abstract class StreamingDataTransform : DataTransform {
    final override fun transform(dataSet: DataSet): DataSet {
        val obsList = StreamSupport.stream(dataSet.spliterator(), false).map { transform(it) }.toList()
        return IterableDataSet(obsList)
    }

    abstract fun transform(observation: Observation): Observation
}
