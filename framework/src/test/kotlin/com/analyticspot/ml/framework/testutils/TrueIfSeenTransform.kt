package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.datatransform.TargetSupervisedLearningTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.observation.SingleValueObservation
import com.fasterxml.jackson.annotation.JsonCreator
import java.util.concurrent.CompletableFuture

/**
 * A very simple [SupervisedLearningTransform] that learns to predict a boolean target based on a single feature of type
 * `String`. During training it simply keeps track of all the unique values it has seen for the feature when the target
 * is `true` in [wordsForTrue]. To make a prediction it predicts `true` if and only if the observed value for the
 * feature in in [wordsForTrue].
 */
class TrueIfSeenTransform : TargetSupervisedLearningTransform<Boolean> {
    override val description: TransformDescription
    val wordsForTrue = mutableSetOf<String>()
    val srcToken: ValueToken<String>
    val resultId: ValueId<Boolean>

    constructor(srcToken: ValueToken<String>, targetToken: ValueToken<Boolean>?,
            resultId: ValueId<Boolean>) : super(targetToken) {
        this.srcToken = srcToken
        this.resultId = resultId
        description = TransformDescription(listOf(ValueToken(resultId)))
    }

    @JsonCreator
    constructor(srcToken: ValueToken<String>, resultId: ValueId<Boolean>): this(srcToken, null, resultId)


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
