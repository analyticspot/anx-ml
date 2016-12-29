package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import java.util.concurrent.CompletableFuture

/**
 * A [DataTransform] that learns from the data. To use it one should call `trainTransform` to both train the algorithm
 * and apply it to the input data. Once trained you can call [transform] to apply it to new, previously unseen data.
 */
interface LearningTransform : DataTransform {
    /**
     * Learn from the data and then applies what was learned to produce a new [DataSet].
     */
    fun trainTransform(dataSet: DataSet): CompletableFuture<DataSet>
}
