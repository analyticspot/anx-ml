package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.SingleDataTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import java.util.ArrayList
import java.util.concurrent.CompletableFuture

/**
 * This transform returns true if the values for all the source columns passed to the contructor are true, false
 * otherwise.
 */
class AndTransform(val srcCols: List<ColumnId<Boolean>>, val resultId: ColumnId<Boolean>) : SingleDataTransform {
    override val description: TransformDescription = TransformDescription(listOf(resultId))

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val resultList = ArrayList<Boolean>(dataSet.numRows)
        for (rowIdx in 0 until dataSet.numRows) {
            var isTrue = true
            for (colId in srcCols) {
                val colVal = dataSet.value(colId, rowIdx)
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
