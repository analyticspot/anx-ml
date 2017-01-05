package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.observation.Observation
import java.util.concurrent.CompletableFuture

/**
 * An abstract base class for [DataTransform] that allows users to override a method that takes a single [Observation]
 * and returns a single [Observation] instead of having to worry about entire [DataSet]s.
 */
abstract class StreamingDataTransform : SingleDataTransform {
    final override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val obsList = dataSet.map { transform(it) }
        return CompletableFuture.completedFuture(IterableDataSet(obsList))
    }

    abstract fun transform(observation: Observation): Observation
}
