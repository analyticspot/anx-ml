package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.utils.isAssignableFrom
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.slf4j.LoggerFactory
import java.util.ArrayList
import java.util.Collections
import java.util.Random

/**
 * DeepLearning4j requires a `MultiDataSetIterator` in order to work with nets that can have multiple inputs and
 * outputs (see https://deeplearning4j.org/compgraph#multidataset-and-the-multidatasetiterator). Unfortunately, they
 * all the implementations of that interface they provide simply read allData from files or some kind but we have our
 * allData in memory and want to use it from there. We also want to be able to convert one or more [DataSet] instances into
 * a `MultiDataSetIterator`. This class allows us to do that.
 *
 * Dl4j also relies on the iterator to provide the batches for stochastic gradient descent (SGD) so we want to be able
 * to convert a big [DataSet] into many smaller batches, each of which is a `MultiDataSet`. Furthermore, we'd like to
 * shuffle all the allData into new/different batches when [reset] is called so that it's useful for SGD. This also handles
 * that for us.
 *
 * To use, pass in a list of allData sets. Each allData set will be one set of features that can be used as input to our net.
 * You also pass a single allData set that contains multiple columns to serve as the targets, one target per column. The
 * target allData sets should be integers in the range [0, numTargetValues) and they will be one-hot encoded by this
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
internal class RandomizingMultiDataSetIterator : MultiDataSetIterator {
    private val allData: MultiDataSet
    private val numRows: Int
        get() = allData.features[0].rows()

    private val batchSize: Int
    private val rng: Random
    // The next batch will contain rows [nextBatchStartIdx, nextBatchStartIdx + batchSize) (or fewer if we have to
    // truncate because the final batch doesn't contain enough rows).
    private var nextBatchStartIdx = 0

    constructor(batchSize: Int, subsets: List<DataSet>, targets: DataSet, rng: Random = Random())
            : this(batchSize, Utils.toMultiDataSet(subsets, targets), rng) {
    }

    constructor(batchSize: Int, srcData: DataSet, featureSubsets: List<List<ColumnId<*>>>,
            targetCols: List<ColumnId<Int>>, rng: Random = Random())
            : this(batchSize, Utils.toMultiDataSet(srcData, featureSubsets, targetCols), rng) {
    }

    private constructor(batchSize: Int, allData: MultiDataSet, rng: Random) {
        this.batchSize = batchSize
        this.allData = allData
        this.rng = rng
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
