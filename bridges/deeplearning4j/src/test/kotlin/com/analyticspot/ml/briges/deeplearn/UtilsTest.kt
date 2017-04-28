package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.util.Random

class UtilsTest : Dl4jTestBase() {
    companion object {
        private val log = LoggerFactory.getLogger(UtilsTest::class.java)
    }

    @Test
    fun testToMultiDataSetFromSingleDs() {
        val c1 = ColumnId.create<Int>("c1")
        val c2 = ColumnId.create<Double>("c2")
        val c3 = ColumnId.create<Float>("c3")
        val c4 = ColumnId.create<Int>("c4")

        val ds = DataSet.build {
            addColumn(c1, listOf(0, 1, 2))
            addColumn(c2, listOf(4.0, 5.0, 6.0))
            addColumn(c3, listOf(7.0f, 8.0f, 9.0f))
            addColumn(c4, listOf(2, 1, 0))
        }

        // We should be able to re-use columns in different feature sets (e.g. to pass the same values to different
        // Neural net inputs) and even re-use them as targets (e.g. to train an embedding).
        val fs1: List<ColumnId<out Number>> = listOf(c1, c2)
        val fs2: List<ColumnId<out Number>> = listOf(c2)
        val fs3: List<ColumnId<out Number>> = listOf(c2, c4)
        val fs4: List<ColumnId<out Number>> = listOf(c2, c3, c4)

        val targs = listOf(c1 to 3, c4 to 3)

        val multiDs = Utils.toMultiDataSet(ds, listOf(fs1, fs2, fs3, fs4), targs)

        assertThat(multiDs.features.size).isEqualTo(4)
        assertThat(multiDs.features[0].shape()).isEqualTo(arrayOf(3, 2))
        assertThat(multiDs.features[1].shape()).isEqualTo(arrayOf(3, 1))
        assertThat(multiDs.features[2].shape()).isEqualTo(arrayOf(3, 2))
        assertThat(multiDs.features[3].shape()).isEqualTo(arrayOf(3, 3))

        assertDataSetsEqual(multiDs.features[0], ds, fs1)
        assertDataSetsEqual(multiDs.features[1], ds, fs2)
        assertDataSetsEqual(multiDs.features[2], ds, fs3)
        assertDataSetsEqual(multiDs.features[3], ds, fs4)

        assertOneHotEqual(multiDs.labels[0], ds.column(c1))
        assertOneHotEqual(multiDs.labels[1], ds.column(c4))
    }

    @Test
    fun testSubsetRows() {
        val c1 = ColumnId.create<Int>("c1")
        val c2 = ColumnId.create<Double>("c2")
        val c3 = ColumnId.create<Float>("c3")
        val c4 = ColumnId.create<Int>("c4")

        val ds = DataSet.build {
            addColumn(c1, listOf(0, 1, 2))
            addColumn(c2, listOf(4.0, 5.0, 6.0))
            addColumn(c3, listOf(7.0f, 8.0f, 9.0f))
            addColumn(c4, listOf(2, 1, 0))
        }

        // We should be able to re-use columns in different feature sets (e.g. to pass the same values to different
        // Neural net inputs) and even re-use them as targets (e.g. to train an embedding).
        val fs1: List<ColumnId<out Number>> = listOf(c1, c2)
        val fs2: List<ColumnId<out Number>> = listOf(c2)

        val targs = listOf(c1 to 3, c4 to 3)

        val multiDs = Utils.toMultiDataSet(ds, listOf(fs1, fs2), targs)

        val sub1 = Utils.subsetRows(multiDs, 0, 1)

        assertDataSetsEqual(sub1.features[0], ds.rowsSubset(setOf(0)), fs1)
        assertDataSetsEqual(sub1.features[1], ds.rowsSubset(setOf(0)), fs2)
        assertOneHotEqual(sub1.labels[0], ds.rowsSubset(setOf(0)).column(c1))
        assertOneHotEqual(sub1.labels[1], ds.rowsSubset(setOf(0)).column(c4))

        val sub2 = Utils.subsetRows(multiDs, 1, 3)

        assertDataSetsEqual(sub2.features[0], ds.rowsSubset(setOf(1, 2)), fs1)
        assertDataSetsEqual(sub2.features[1], ds.rowsSubset(setOf(1, 2)), fs2)
        assertOneHotEqual(sub2.labels[0], ds.rowsSubset(setOf(1, 2)).column(c1))
        assertOneHotEqual(sub2.labels[1], ds.rowsSubset(setOf(1, 2)).column(c4))
    }

    @Test
    fun testShuffle() {
        val mdsFeatures = arrayOf(
                // A 3x2 matrix where the values in each row are identical and the values in different rows are
                // different. Here row # == value
                Nd4j.create(arrayOf(
                        doubleArrayOf(0.0, 0.0),
                        doubleArrayOf(1.0, 1.0),
                        doubleArrayOf(2.0, 2.0))),
                // A 3x1 matrix where row # = value
                Nd4j.create(arrayOf(
                        doubleArrayOf(0.0),
                        doubleArrayOf(1.0),
                        doubleArrayOf(2.0)
                ))
        )

        val targets = arrayOf(
                // One-hot encoded targets where the hot target == the row
                Nd4j.create(arrayOf(
                        doubleArrayOf(1.0, 0.0, 0.0),
                        doubleArrayOf(0.0, 1.0, 0.0),
                        doubleArrayOf(0.0, 0.0, 1.0)
                ))
        )
        val mds = MultiDataSet(mdsFeatures, targets)

        // Pass a seed here so we can be sure the shuffle actually changes the row ordering. Since there's only 3 rows
        // there's a decent chance that wouldn't happen otherwise.
        Utils.shuffle(mds, Random(111))

        // Make sure things were actually shuffled
        assertThat(mds.features[0].getColumn(0))
                .isNotEqualTo(Nd4j.create(doubleArrayOf(0.0, 1.0, 2.0), intArrayOf(3, 1)))

        // Make sure all the rows are still in the same order and that all sub-data sets were shuffled the same way
        for (i in 0..2) {
            assertThat(mds.features[0].getDouble(i, 0)).isEqualTo(mds.features[0].getDouble(i, 1))
            val originalRow = mds.features[0].getDouble(i, 0)
            assertThat(mds.features[1].getDouble(i, 0)).isEqualTo(originalRow)
            for (j in 0..2) {
                val origRowIdx = originalRow.toInt()
                if (j == origRowIdx) {
                    assertThat(mds.labels[0].getDouble(i, j)).isEqualTo(1.0)
                } else {
                    assertThat(mds.labels[0].getDouble(i, j)).isEqualTo(0.0)
                }
            }
        }
    }

    private fun assertDataSetsEqual(nd4jData: INDArray, ds: DataSet, cols: List<ColumnId<out Number>>) {
        cols.forEachIndexed { colIdx, columnId ->
            ds.column(columnId).forEachIndexed { rowIdx, value ->
                assertThat(nd4jData.getDouble(rowIdx, colIdx)).isEqualTo(value?.toDouble())
            }
        }
    }

    private fun assertOneHotEqual(nd4jData: INDArray, targets: Iterable<Int?>) {
        assertThat(targets).hasSize(nd4jData.rows())
        targets.forEachIndexed { rowIdx, targetValue ->
            assertThat(targetValue).isNotNull()
            log.debug("Checking that {} one-hot encodes {}", nd4jData.getRow(rowIdx), targetValue)
            for (colIdx in 0.until(nd4jData.columns())) {
                if (colIdx == targetValue) {
                    assertThat(nd4jData.getDouble(rowIdx, colIdx)).isEqualTo(1.0)
                } else {
                    assertThat(nd4jData.getDouble(rowIdx, colIdx)).isEqualTo(0.0)
                }
            }
        }
    }
}
