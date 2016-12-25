package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet

/**
 *
 */
interface LearningTransform : DataTransform {
    fun train(dataSets: DataSet)

    fun trainTransform(dataSets: DataSet): DataSet
}
