package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.utils.isAssignableFrom
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import java.util.ArrayList
import java.util.Collections
import java.util.Random

/**
 * DeepLearning4j requires a `MultiDataSetIterator` in order to work with nets that can have multiple inputs and
 * outputs (see https://deeplearning4j.org/compgraph#multidataset-and-the-multidatasetiterator). Unfortunately, they
 * all the implementations of that interface they provide simply read data from files or some kind but we have our
 * data in memory and want to use it from there. We also want to be able to convert one or more [DataSet] instances into
 * a `MultiDataSetIterator`. This class allows us to do that.
 *
 * Dl4j also relies on the iterator to provide the batches for stochastic gradient descent (SGD) so we want to be able
 * to convert a big [DataSet] into many smaller batches, each of which is a `MultiDataSet`. Furthermore, we'd like to
 * shuffle all the data into new/different batches when [reset] is called so that it's useful for SGD. This also handles
 * that for us.
 *
 * To use, pass in a list of data sets. Each data set will be one set of features that can be used as input to our net.
 * You also pass a single data set that contains multiple columns to serve as the targets, one target per column. The
 * target data sets should be integers in the range [0, numTargetValues) and they will be one-hot encoded by this
 * class.
 *
 * @param batchSize the number of items in each mini-batch
 * @param subsets a list of [DataSet] instances. Each [DataSet] will correspond to a single set of inputs to the
 *     network. As the network requires doubles as inputs all columns must be of `java.lang.Number` type (e.g. `String`
 *     columns should be one-hot encoded before being used here.)
 * @param targets a [DataSet] that contains only integer columns. There will be one target for each column here but
 *     the target will be one-hot encoded.
 * @param rng: The random number generator to use to shuffle the rows.
 */
internal class RandomizingMultiDataSetIterator(val batchSize: Int,
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
        require(subsets.all { subset -> subset.columnIds.all { Number::class isAssignableFrom it.clazz } }) {
            "All columns must be of type java.lang.Number (or a subclass or it)."
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
            @Suppress("UNCHECKED_CAST")
            val colId = targets.columnIds[colIdx] as ColumnId<Int>
            // We add 1 here because we assume the targets are 0 indexed - thus the number of target values is 1 more
            // than the largest observed target.
            targets.column(colId).asSequence().maxBy { it!! }!! + 1
        }
        log.debug("Target sizes are: {}", targetSizes)

        batchIndices = ArrayList(numRows)
        for (idx in 0 until numRows) {
            batchIndices.add(idx)
        }

        reset()
    }

    companion object {
        private val log = LoggerFactory.getLogger(RandomizingMultiDataSetIterator::class.java)
    }

    override fun next(num: Int): MultiDataSet {
        check(hasNext())
        val batchEndIdx = Math.min(nextBatchStartIdx + num, numRows)
        val numInThisBatch = batchEndIdx - nextBatchStartIdx
        val batchFeatures = Array<INDArray>(subsets.size) { idx ->
            Nd4j.zeros(numInThisBatch, subsets[idx].numColumns)
        }
        val batchTargets = Array<INDArray>(subsets.size) { targIdx ->
            Nd4j.zeros(numInThisBatch, targetSizes[targIdx])
        }

        // Lots of indexes here:
        // i: 0 to number of items in this batch
        // batchIdxIdx: Index into batchIdx that lets us look up the shuffled/randomized index into the data sets from
        //     which we should pull data for this batch.
        // dataRowIdx: the index we found in batchIdx. This is the row in the underlying data we'll use
        // subsetIdx: a MultiDataSet is composed of multiple subsets of data. This is the subset we're currently
        //     working on
        // colIdx: this is the column in the current subset we're working on
        // targetIdx: a MultiDataSet is also composed of multiple targets. This is the target we're working on
        for (i in 0 until numInThisBatch) {
            val batchIdxIdx = nextBatchStartIdx + i
            val dataRowIdx = batchIndices[batchIdxIdx]
            for (subsetIdx in subsets.indices) {
                val subset = subsets[subsetIdx]
                for (colIdx in subset.columnIds.indices) {
                    val curValue: Number = subset.value(dataRowIdx, subset.columnIds[colIdx]) as Number
                    batchFeatures[subsetIdx].put(i, colIdx, curValue)
                }
            }

            for (targetIdx in targets.columnIds.indices) {
                @Suppress("UNCHECKED_CAST")
                val curColId = targets.columnIds[targetIdx] as ColumnId<Int>
                val curValue: Int = targets.value(dataRowIdx, curColId)!!
                // Put a 1 in the column corresponding to curValue so we can 1-hot encode the correct value
                batchTargets[targetIdx].put(i, curValue, 1)
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
        nextBatchStartIdx = 0
    }

    override fun hasNext(): Boolean {
        return nextBatchStartIdx < numRows
    }

    override fun asyncSupported(): Boolean = false
}
