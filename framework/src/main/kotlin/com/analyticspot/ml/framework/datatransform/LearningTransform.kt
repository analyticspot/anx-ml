package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet

/**
 * A [DataTransform] that learns from the data. To use it one should call `train` or `trainTransform` before calling
 * `transform`. The `train` and `trainTransform` methods allow the algorithm to learn from the data. The `transform`
 * method that applies the trained transformation.
 */
interface LearningTransform : DataTransform {
    /**
     * Learn from the given data sets.
     */
    fun train(dataSet: DataSet)

    /**
     * Has the same effect as calling `train` and then calling `transform`. Howver, for some algorithms it can be more
     * efficient to combine these steps.
     */
    fun trainTransform(dataSet: DataSet): DataSet
}
