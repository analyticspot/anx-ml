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
import java.util.ArrayList
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * This transform returns true if the values for all the source columns passed to the contructor are true, false
 * otherwise.
 */
class AndTransform(val srcCols: List<ColumnId<Boolean>>, val resultId: ColumnId<Boolean>) : SingleDataTransform {
    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val resultList = ArrayList<Boolean>(dataSet.numRows)
        for (rowIdx in 0 until dataSet.numRows) {
            var isTrue = true
            for (colId in srcCols) {
                val colVal = dataSet.value(rowIdx, colId)
                if (colVal == null || !colVal) {
                    isTrue = false
                    break
                }
            }
            resultList.add(isTrue)
        }
        return CompletableFuture.completedFuture(DataSet.create(resultId, resultList))
    }
}
