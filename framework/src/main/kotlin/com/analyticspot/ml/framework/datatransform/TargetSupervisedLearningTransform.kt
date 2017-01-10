package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ValueToken
import com.fasterxml.jackson.annotation.JsonIgnore
import java.util.concurrent.CompletableFuture

/**
 * As supervised learning algorithms that work with a single target value are common this is a convenience class that
 * extracts the target for the user. Subclasses then implmement the [trainTransform] overload that takes a `TargetT`
 * rather than the one that takes an entire [DataSet].
 *
 * @param targetToken the token used to extract the target value from the data. Can be null as it will not be available
 *      when deserializing a trained transform.
 * @param <TargetT> the type of the target.
 */
abstract class TargetSupervisedLearningTransform<TargetT>(@JsonIgnore val targetToken: ValueToken<TargetT>?)
    : SupervisedLearningTransform {
    override fun trainTransform(dataSet: DataSet, trainDs: DataSet): CompletableFuture<DataSet> {
        if (targetToken != null) {
            val targets = trainDs.map { it.value(targetToken) }
            return trainTransform(dataSet, targets)
        } else {
            throw IllegalStateException("The target token must be available when training but it is null")
        }
    }

    abstract fun trainTransform(dataSet: DataSet, target: Iterable<TargetT>): CompletableFuture<DataSet>

}
