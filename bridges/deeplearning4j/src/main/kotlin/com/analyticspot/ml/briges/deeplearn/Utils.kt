package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.slf4j.LoggerFactory
import java.util.Random

/**
 * Various utilities for working with DeepLearning4j.
 */
object Utils {
    /**
     * Creates a `MultiDataSet` from a [DataSet] and some information specifying which columns map to which feature
     * subsets, which map to which target groups, etc.
     *
     * @param srcDs the data set holding the data for the MultiDataSet
     * @param featureSubsets a list of lists such that featureSubsets[i] is the list of columns whose data belongs in
     *     the i^th group of feature inputs.
     * @param targetCols a list of columns that hold the targets. Each target must be an integer in the range
     *     `[0, numTargets]`. These will be 1-hot encoded.
     * @param targetSizes `targetSizes[i]` is the number of distinct values in the data corresponding to
     *     `targetCols[i]`.
     */
    fun toMultiDataSet(srcDs: DataSet, featureSubsets: List<List<ColumnId<*>>>,
            targetCols: List<ColumnId<Int>>, targetSizes: List<Int>): MultiDataSet {
        require(targetSizes.size == targetCols.size)
        val featureArrays = Array<INDArray>(featureSubsets.size) { idx ->
            val indArrayData = Nd4j.zeros(srcDs.numRows, featureSubsets[idx].size)

            featureSubsets[idx].forEachIndexed { colIdx, columnId ->
                srcDs.column(columnId).forEachIndexed { rowIdx, colValueAsAny ->
                    indArrayData.put(rowIdx, colIdx, colValueAsAny as Number)
                }
            }

            indArrayData
        }

        val targetArrays = Array<INDArray>(targetCols.size) { idx ->
            val targetCol = targetCols[idx]
            val targetSize = targetSizes[idx]

            val indTargetMatrix = Nd4j.zeros(srcDs.numRows, targetSize)

            srcDs.column(targetCol).forEachIndexed { rowIdx, targetValue ->
                check(targetValue!! < targetSize)
                indTargetMatrix.put(rowIdx, targetValue, 1)
            }

            indTargetMatrix
        }

        // Yes, Nd4j really did give the class the same name as the interface!
        return org.nd4j.linalg.dataset.MultiDataSet(featureArrays, targetArrays)
    }

    /**
     * Like the other [toMultiDataSet] overload but the subsets are defined by a list of [DataSet] instances and the
     * targets are defined by all the columns in another data set.
     */
    fun toMultiDataSet(featureSubsets: List<DataSet>, targets: DataSet, targetSizes: List<Int>): MultiDataSet {
        require(targetSizes.size == targets.numColumns)
        val combinedDs = featureSubsets.reduce { ds1, ds2 -> ds1.combineAndSkipDuplicates(ds2) }
                .combineAndSkipDuplicates(targets)

        val featureCols = featureSubsets.map { featureDs ->
            featureDs.columnIds.toList()
        }
        @Suppress("UNCHECKED_CAST")
        val targetCols = targets.columnIds.toList() as List<ColumnId<Int>>

        return toMultiDataSet(combinedDs, featureCols, targetCols, targetSizes)
    }

    /**
     * Returns a new `MultiDataSet` that contains just rows `[start, end)` of the `src`. The returned `MultiDataSet`
     * will be just a view of `src` so this is 0-copy, but modifications to one will be reflected in the other.
     */
    fun subsetRows(src: MultiDataSet, start: Int, end: Int): MultiDataSet {
        // There isn't a split or slice method on MultiDataSet but we can slice all the ndarrays in the same way and
        // construct a new MutliDataSet pretty easily so that's what we do here. Note that these slice operations
        // produce views and so are 0-copy.
        val batchFeatures: Array<INDArray> = src.features.map {
            it.get(NDArrayIndex.interval(start, end), NDArrayIndex.all())
        }.toTypedArray()

        val batchTargets: Array<INDArray> = src.labels.map {
            it.get(NDArrayIndex.interval(start, end), NDArrayIndex.all())
        }.toTypedArray()


        return org.nd4j.linalg.dataset.MultiDataSet(batchFeatures, batchTargets)
    }

    /**
     * An extension method that allows us to shuffle the rows of a `MultiDataSet`.
     *
     * TODO(NOW): What happens to other views???
     */
    fun shuffle(data: MultiDataSet, rng: Random = Random()) {
        if (data.features[0].rows() == 1) {
            // There's only 1 row so a shuffle doesn't do anything. But Nd4j.shuffle crashes in that case so handle it
            // explicitly
            return
        }

        val arraysToShuffle: List<INDArray> = data.features.toList() + data.labels.toList()
        // We always shuffle along dimension 1 (e.g. the rows)
        val dims = arraysToShuffle.map { intArrayOf(1) }
        Nd4j.shuffle(arraysToShuffle, rng, dims)
    }
}
