package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import com.analyticspot.ml.framework.description.TransformDescription
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.SingleValueObservation
import java.util.concurrent.CompletableFuture
import java.util.concurrent.atomic.AtomicInteger

/**
 * Silly transform that inverts the target value. The purpose here is (1) to create a new data set that is
 * only for training and (2) to be able to check that the transform is run when training but not transforming.
 */
class InvertBoolean(private val srcToken: ValueToken<Boolean>,
        private val resultId: ValueId<Boolean>) : SingleDataTransform {
    val numCalls = AtomicInteger(0)

    override val description: TransformDescription = TransformDescription(listOf(ValueToken(resultId)))

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        numCalls.incrementAndGet()
        val newObs = dataSet.map { !it.value(srcToken) }.map { SingleValueObservation.create(it) }
        return CompletableFuture.completedFuture(IterableDataSet(newObs))
    }
}
