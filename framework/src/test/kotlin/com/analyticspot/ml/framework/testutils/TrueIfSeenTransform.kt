package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.TargetSupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import com.fasterxml.jackson.annotation.JsonCreator
import java.util.concurrent.CompletableFuture

/**
 * A very simple [SupervisedLearningTransform] that learns to predict a boolean target based on a single feature of type
 * `String`. During training it simply keeps track of all the unique values it has seen for the feature when the target
 * is `true` in [wordsForTrue]. To make a prediction it predicts `true` if and only if the observed value for the
 * feature in in [wordsForTrue].
 */
class TrueIfSeenTransform(
        val srcColumn: ColumnId<String>, targetColumn: ColumnId<Boolean>?, val resultId: ColumnId<Boolean>)
    : TargetSupervisedLearningTransform<Boolean>(targetColumn) {
    override val description: TransformDescription = TransformDescription(listOf(resultId))
    val wordsForTrue = mutableSetOf<String>()

    @JsonCreator
    private constructor(srcColumn: ColumnId<String>, resultId: ColumnId<Boolean>): this(srcColumn, null, resultId)

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val resultList = dataSet.column(srcColumn)
                .map { wordsForTrue.contains(it) }
                .toList()
        return CompletableFuture.completedFuture(DataSet.create(resultId, resultList))
    }

    override fun trainTransform(dataSet: DataSet, target: Column<Boolean?>): CompletableFuture<DataSet> {
        dataSet.column(srcColumn).zip(target).forEach { valueTargetPair ->
            val (value, targ) = valueTargetPair
            if (targ!! && value != null) {
                wordsForTrue.add(value)
            }
        }
        return transform(dataSet)
    }
}
