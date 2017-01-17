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

/**
 * A [DataTransform] that learns from the data. To use it one should call `trainTransform` to both train the algorithm
 * and apply it to the input data. Once trained you can call [transform] to apply it to new, previously unseen data.
 */
interface LearningTransform : SingleDataTransform {
    /**
     * Learn from the data and then applies what was learned to produce a new [DataSet].
     */
    fun trainTransform(dataSet: DataSet): CompletableFuture<DataSet>
}
