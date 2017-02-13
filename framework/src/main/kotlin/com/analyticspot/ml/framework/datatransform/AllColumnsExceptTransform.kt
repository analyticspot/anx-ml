package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * Like [ColumnSubsetTransform] except it keeps all the columns in the source data set except those specified.
 */
class AllColumnsExceptTransform(val toExclude: Set<String>) : SingleDataTransform {
    companion object {
        private val log = LoggerFactory.getLogger(AllColumnsExceptTransform::class.java)

        /**
         * Create a [AllColumnsExceptTransform] that keeps all columns except those passed here.
         */
        fun create(vararg exclude: String) = AllColumnsExceptTransform(exclude.toSet())
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        return CompletableFuture.completedFuture(DataSet.build {
            dataSet.columnIds.forEach {
                if (it.name in toExclude) {
                    log.debug("Dropping column {}", it)
                } else {
                    val md = dataSet.metaData[it.name]
                    @Suppress("UNCHECKED_CAST")
                    addColumn(it as ColumnId<Any>, dataSet.column(it), md)
                }
            }
        })
    }
}
