package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.nd4j.linalg.api.buffer.DataBuffer.Type.DOUBLE
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.IsNaN
import org.nd4j.linalg.indexing.conditions.Not
import org.slf4j.LoggerFactory
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
 */
internal class RandomizingMultiDataSetIterator : MultiDataSetIterator {
    private val allData: MultiDataSet
    private val numRows: Int
        get() = allData.features[0].rows()

    private val batchSize: Int
    private val rng: Random
    // The next batch will contain rows [nextBatchStartIdx, nextBatchStartIdx + batchSize) (or fewer if we have to
    // truncate because the final batch doesn't contain enough rows).
    private var nextBatchStartIdx = 0

    /**
     * Constructor.
     *
     * @param batchSize the number of items in each mini-batch
     * @param subsets a list of [DataSet] instances. Each [DataSet] will correspond to a single set of inputs to the
     *     network. As the network requires doubles as inputs all columns must be of `java.lang.Number` type (e.g. `String`
     *     columns should be one-hot encoded before being used here.)
     * @param targets a [DataSet] that contains only integer columns. There will be one target for each column here but
     *     the target will be one-hot encoded.
     * @param targetSizes indicates how many distinct target values are in each column of the targets [DataSet]. The
     *    key is the name of a column in `targets` and the value is the number of values that column can take on.
     * @param rng: The random number generator to use to shuffle the rows.
     */
    constructor(batchSize: Int, subsets: List<DataSet>, targets: DataSet,
            targetSizes: Map<String, Int>, rng: Random = Random())
            : this(batchSize, Utils.toMultiDataSet(subsets, targets, targetSizes), rng) {
    }

    /**
     * Constructor.
     *
     * @param batchSize the number of items in each mini-batch
     * @param srcData the [DataSet] from which all feature and target values will be drawn.
     * @param featureSubsets a list of [ColumnId] instances. Each set of [ColumnId]s will correspond to a single set of
     *     inputs to the network. As the network requires doubles as inputs all columns must be of `java.lang.Number`
     *     type (e.g. `String` columns should be one-hot encoded before being used here.)
     * @param targetCols a list of [ColumnId] from which the targets should be extracted. There will be one target for
     *     each column here but the target will be one-hot encoded. The 2nd part of the `Pair` is the number of unique
     *     values for the given target column (so we know how many columns to create in our 1-hot encoding).
     * @param rng: The random number generator to use to shuffle the rows.
     */
    constructor(batchSize: Int, srcData: DataSet, featureSubsets: List<List<ColumnId<*>>>,
            targetCols: List<Pair<ColumnId<Int>, Int>>, rng: Random = Random())
            : this(batchSize, Utils.toMultiDataSet(srcData, featureSubsets, targetCols), rng) {
    }

    private constructor(batchSize: Int, allData: MultiDataSet, rng: Random) {
        this.batchSize = batchSize
        this.allData = allData
        this.rng = rng

        allData.features.forEachIndexed { idx, indArray ->
            check(indArray.data().dataType() == DOUBLE) {
                "Expected input $idx to be an INDArray of double but it was ${indArray.data().dataType()}. " +
                        "Note that ND4j requires this to be set globally via" +
                        "DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)"
            }
            check(BooleanIndexing.and(indArray, Not(IsNaN()))) {
                "Input $idx contains some NaN values"
            }
        }

        allData.labels.forEachIndexed { idx, indArray ->
            check(indArray.data().dataType() == DOUBLE) {
                "Expected output/targets $idx to be an INDArray of double but it was ${indArray.data().dataType()}. " +
                "Note that ND4j requires this to be set globally via" +
                        "DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)"
            }
            check(BooleanIndexing.and(indArray, Not(IsNaN()))) {
                "Output $idx contains some NaN values"
            }
        }

        reset()
    }

    companion object {
        private val log = LoggerFactory.getLogger(RandomizingMultiDataSetIterator::class.java)
    }

    override fun next(num: Int): MultiDataSet {
        check(hasNext())
        val batchEndIdx = Math.min(nextBatchStartIdx + num, numRows)

        val resultDs = Utils.subsetRows(allData, nextBatchStartIdx, batchEndIdx)

        nextBatchStartIdx = batchEndIdx
        return resultDs
    }

    override fun next(): MultiDataSet {
        return next(batchSize)
    }

    override fun resetSupported(): Boolean = true

    override fun setPreProcessor(preprocessor: MultiDataSetPreProcessor?) {
        // The semantics of this are very unclear: it is supposed to be run on each mini-batch, even after reset, or run
        // only once on the base data? Is it supposed to modify the underlying data or produce a copy? Since I don't
        // think we currently need this we simply won't support it. If we ever need this we're going to have to either
        // look at nd4j source code or ask some questions online.
        throw UnsupportedOperationException()
    }

    override fun remove() {
        throw UnsupportedOperationException()
    }

    override fun reset() {
        nextBatchStartIdx = 0
        Utils.shuffle(allData, rng)
    }

    override fun hasNext(): Boolean {
        return nextBatchStartIdx < numRows
    }

    override fun asyncSupported(): Boolean = false
}
