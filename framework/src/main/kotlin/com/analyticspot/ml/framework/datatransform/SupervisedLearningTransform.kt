package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import java.util.concurrent.CompletableFuture

/**
 * Like [LearningTransform] but for supervised learning algorithms. The [trainTransform] method, in addition to a
 * [DataSet] also take a second [DataSet] which is required only during training. This second [DataSet] typically
 * contains a target. [SupervisedLearningTransform] is a different class than [LearningTransform] as we want to be able
 * to know which parts of the graph are required only for training so we don't generate that data when calling
 * [transform] to get predictions.
 */
interface SupervisedLearningTransform : SingleDataTransform {
    /**
     * See class comments.
     */
    fun trainTransform(dataSet: DataSet, trainDs: DataSet): CompletableFuture<DataSet>
}
