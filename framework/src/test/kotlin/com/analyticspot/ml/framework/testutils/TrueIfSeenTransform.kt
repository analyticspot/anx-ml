/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * The ANX ML library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.dataset.Column
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.TargetSupervisedLearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import com.fasterxml.jackson.annotation.JsonCreator
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

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

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val resultList = dataSet.column(srcColumn)
                .map { wordsForTrue.contains(it) }
                .toList()
        return CompletableFuture.completedFuture(DataSet.create(resultId, resultList))
    }

    override fun trainTransform(dataSet: DataSet, target: Column<Boolean?>, exec: ExecutorService)
            : CompletableFuture<DataSet> {
        dataSet.column(srcColumn).zip(target).forEach { valueTargetPair ->
            val (value, targ) = valueTargetPair
            if (targ!! && value != null) {
                wordsForTrue.add(value)
            }
        }
        return transform(dataSet, exec)
    }
}
