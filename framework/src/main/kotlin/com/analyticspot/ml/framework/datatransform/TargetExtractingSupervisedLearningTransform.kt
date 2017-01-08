package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ValueToken
import java.util.concurrent.CompletableFuture

/**
 * As supervised learning algorithms that work with a single target value are common this is a convenience class that
 * extracts the target for the user. Subclasses then implmement the [trainTransform] overload that takes a `TargetT`
 * rather than the one that takes an entire [DataSet].
 *
 * @param <TargetT> the type of the target.
 */
abstract class TargetExtractingSupervisedLearningTransform<TargetT>(private val targetToken: ValueToken<TargetT>)
    : SupervisedLearningTransform {
    override fun trainTransform(dataSet: DataSet, trainDs: DataSet): CompletableFuture<DataSet> {
        val targets = trainDs.map { it.value(targetToken) }
        return trainTransform(dataSet, targets)
    }

    abstract fun trainTransform(dataSet: DataSet, target: Iterable<TargetT>): CompletableFuture<DataSet>

}
