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

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import java.util.concurrent.CompletableFuture
import java.util.concurrent.atomic.AtomicInteger

/**
 * Silly transform that inverts the target value. The purpose here is (1) to create a new data set that is
 * only for training and (2) to be able to check that the transform is run when training but not transforming.
 */
class InvertBoolean(private val srcColumn: ColumnId<Boolean>,
        private val resultId: ColumnId<Boolean>) : SingleDataTransform {
    val numCalls = AtomicInteger(0)

    override val description: TransformDescription = TransformDescription(listOf(resultId))

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        numCalls.incrementAndGet()
        val newCol = dataSet.column(srcColumn).mapToColumn { !it!! }
        return CompletableFuture.completedFuture(DataSet.Companion.create(resultId, newCol))
    }
}
