package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.ListColumn
import com.analyticspot.ml.framework.datatransform.LearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import org.slf4j.LoggerFactory
import java.util.Collections
import java.util.concurrent.CompletableFuture

/**
 * Learns the minimum value for a single column and produces a [DataSet] with a single column, identified by `resultId`,
 * which contains that minimum value.
 */
class LearnMinTransform(private val srcColumn: ColumnId<Int>, val resultId: ColumnId<Int>) : LearningTransform {
    private var minValue: Int = Int.MAX_VALUE
    override val description: TransformDescription
        get() = TransformDescription(listOf(resultId))

    companion object {
        private val log = LoggerFactory.getLogger(LearnMinTransform::class.java)
    }

    override fun trainTransform(dataSet: DataSet): CompletableFuture<DataSet> {
        log.debug("{} is training", this.javaClass)
        val dsMin = dataSet.column(srcColumn).map { it ?: Int.MAX_VALUE }.min()
        minValue = dsMin ?: minValue
        return transform(dataSet)
    }

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val col = ListColumn(Collections.nCopies(dataSet.numRows, minValue))
        val ds = DataSet.build {
            addColumn(resultId, col)
        }
        return CompletableFuture.completedFuture(ds)
    }
}
