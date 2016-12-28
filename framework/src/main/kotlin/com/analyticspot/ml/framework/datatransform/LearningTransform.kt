package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet

/**
 * A [DataTransform] that learns from the data. To use it one should call `train` or `trainTransform` before calling
 * `execute`. The `train` and `trainTransform` methods allow the algorithm to learn from the data. The `execute`
 * method that applies the trained transformation.
 */
interface LearningTransform : DataTransform {
    /**
     * Learn from the data and then applies what was learned to produce a new [DataSet].
     */
    fun trainTransform(dataSet: DataSet): DataSet
}
