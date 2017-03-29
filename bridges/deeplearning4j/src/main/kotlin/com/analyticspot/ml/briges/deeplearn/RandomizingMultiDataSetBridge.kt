package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.utils.isAssignableFrom
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.util.ArrayList
import java.util.Collections
import java.util.Random

/**
 *
 */
internal class RandomizingMultiDataSetBridge(val batchSize: Int,
        val subsets: List<DataSet>, val targets: DataSet,
        val rng: Random = Random()) : MultiDataSetIterator {
    // An array of indices into the data (subsets and targets). This is randomly shuffled on each call to reset so that
    // our batches are random.
    private val batchIndices: ArrayList<Int>
    private val numRows: Int
    private var nextBatchStartIdx = 0
    // We have to 1-hot encode all the targets. The targets are integers indicating the unique target value so this
    // tells us the largest value for each target so we know how big the 1-hot array needs to be.
    private val targetSizes: Array<Int>

    private var preprocessor: MultiDataSetPreProcessor? = null

    init {
        require(subsets.size > 0) {
            "There must be at least one subset of data."
        }
        numRows = subsets[0].numRows
        require(numRows > 0) {
            "Data set is empty."
        }

        require(subsets.all { it.numRows == numRows }) {
            "All subsets must have the same number of rows."
        }
        require(subsets.all { subset -> subset.columnIds.all { Double::class isAssignableFrom it.clazz } }) {
            "It must be possible to assign all columns to a Double."
        }

        require(targets.numRows == numRows) {
            "The targets must have exactly one entry per row in the subsets."
        }
        require(targets.columnIds.all { Int::class isAssignableFrom it.clazz }) {
            "The targets must be integers with each unique value indicating a target class."
        }
        require(targets.columnIds.all { targets.column(it).asSequence().all { it != null } }) {
            "The targets must all be non-null."
        }

        targetSizes = Array<Int>(targets.numColumns) { colIdx ->
            val colId = targets.columnIds[colIdx] as ColumnId<Int>
            targets.column(colId).asSequence().maxBy { it!! }!!
        }

        batchIndices = ArrayList(numRows)
        for (idx in 0 until numRows) {
            batchIndices.add(idx)
        }

        reset()
    }


    override fun next(num: Int): MultiDataSet {
        check(hasNext())
        val batchEndIdx = Math.min(nextBatchStartIdx + num, numRows - 1)
        val batchFeatures = Array<INDArray>(subsets.size) { idx ->
            Nd4j.zeros(numRows, subsets[idx].numColumns)
        }
        val batchTargets = Array<INDArray>(subsets.size) { targIdx ->
            Nd4j.zeros(numRows, targetSizes[targIdx])
        }

        // In this loop rowIdx is the index of the row in the MultiDataSet we'll return. However, since we randomize our
        // batches it is also the index into batchIndices but it is **not** the index into the targets or the subsets.
        // dataRowIdx is the index into the targets and subsets.
        for (rowIdx in nextBatchStartIdx until batchEndIdx) {
            val dataRowIdx = batchIndices[rowIdx]
            for (subsetIdx in subsets.indices) {
                val subset = subsets[subsetIdx]
                for (colIdx in subset.columnIds.indices) {
                    val curValue: Double = subset.value(dataRowIdx, subset.columnIds[colIdx]) as Double
                    batchFeatures[subsetIdx].put(rowIdx, colIdx, curValue)
                }
            }

            for (targetIdx in targets.columnIds.indices) {
                @Suppress("UNCHECKED_CAST")
                val curColId = targets.columnIds[targetIdx] as ColumnId<Int>
                val curValue: Int = targets.value(dataRowIdx, curColId)!!
                // Put a 1 in the column corresponding to curValue so we can 1-hot encode the correct value
                batchTargets[targetIdx].put(rowIdx, curValue, 1)
            }

        }

        nextBatchStartIdx = batchEndIdx
        // Yes, Nd4j really did give the class the same name as the interface!
        val resultDs = org.nd4j.linalg.dataset.MultiDataSet(batchFeatures, batchTargets)
        preprocessor?.preProcess(resultDs)
        return resultDs
    }

    override fun next(): MultiDataSet {
        return next(batchSize)
    }

    override fun resetSupported(): Boolean = true

    override fun setPreProcessor(preprocessor: MultiDataSetPreProcessor?) {
        this.preprocessor = preprocessor
    }

    override fun remove() {
        throw UnsupportedOperationException()
    }

    override fun reset() {
        Collections.shuffle(batchIndices, rng)
    }

    override fun hasNext(): Boolean {
        return nextBatchStartIdx < numRows - 1
    }

    override fun asyncSupported(): Boolean = false
}
