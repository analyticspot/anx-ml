package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet

/**
 * Like [LearningTransform] but for supervised learning algorithms. The [train] and [trainTransform] methods, in
 * addition to a [DataSet] also take a target of type `TargetT`. [SupervisedLearningTransform] is a different class than
 * [LearningTransform] as we want to be able to know which parts of the graph are required only for training so we
 * don't generate that data when calling [transform] to get predictions.
 */
interface SupervisedLearningTransform : DataTransform {
    /**
     * Has the same effect as calling [train] and then calling [transform]. However, for some algorithms it can be more
     * efficient to combine these steps.
     */
    fun trainTransform(dataSet: DataSet, trainDs: DataSet): DataSet
}