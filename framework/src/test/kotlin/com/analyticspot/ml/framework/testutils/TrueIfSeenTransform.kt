package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.datatransform.SupervisedLearningTransform
import com.analyticspot.ml.framework.datatransform.TargetSupervisedLearningTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.SingleValueObservation
import java.util.concurrent.CompletableFuture

/**
 * A very simple [SupervisedLearningTransform] that learns to predict a boolean target based on a single feature of type
 * `String`. During training it simply keeps track of all the unique values it has seen for the feature when the target
 * is `true` in [wordsForTrue]. To make a prediction it predicts `true` if and only if the observed value for the
 * feature in in [wordsForTrue].
 */
class TrueIfSeenTransform(
        val srcToken: ValueToken<String>, targetToken: ValueToken<Boolean>, val resultId: ValueId<Boolean>)
    : TargetSupervisedLearningTransform<Boolean>(targetToken) {
    override val description: TransformDescription = TransformDescription(listOf(ValueToken(resultId)))
    val wordsForTrue = mutableSetOf<String>()

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val resultList = dataSet
                .map { wordsForTrue.contains(it.value(srcToken)) }
                .map { SingleValueObservation.create(it) }
                .toList()
        return CompletableFuture.completedFuture(IterableDataSet(resultList))
    }

    override fun trainTransform(dataSet: DataSet, target: Iterable<Boolean>): CompletableFuture<DataSet> {
        dataSet.zip(target).forEach { obsTargetPair ->
            if (obsTargetPair.second) {
                // The target was true so save the word in wordsForTrue
                wordsForTrue.add(obsTargetPair.first.value(srcToken))
            }
        }

        return transform(dataSet)
    }
}
