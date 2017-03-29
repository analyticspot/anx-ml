package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator

/**
 * Created by oliver on 3/28/17.
 */
class MultiDataSetIteratorBuilder {
    var batchSize: Int? = null
    private val subsets = mutableListOf<DataSet>()

     fun addSubset(dataSet: DataSet): MultiDataSetIteratorBuilder {
         if (subsets.size > 0) {
             require(dataSet.numRows == subsets[0].numRows) {
                 "All subsets must have the same number of rows."
             }
         }
         subsets.add(dataSet)
         return this
     }

    fun withBatchSize(batchSize: Int): MultiDataSetIteratorBuilder {
        require(batchSize > 0) {
            "batch size must be > 0."
        }
        this.batchSize = batchSize
        return this
    }

    fun build(): MultiDataSetIterator {

    }
}
