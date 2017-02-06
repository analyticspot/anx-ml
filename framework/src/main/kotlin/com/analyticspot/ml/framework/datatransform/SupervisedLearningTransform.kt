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

package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

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
    fun trainTransform(dataSet: DataSet, trainDs: DataSet, exec: ExecutorService): CompletableFuture<DataSet>
}
