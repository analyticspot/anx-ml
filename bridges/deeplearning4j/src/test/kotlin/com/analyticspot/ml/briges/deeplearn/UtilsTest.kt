package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.LoggerFactory
import org.testng.annotations.Test

class UtilsTest {
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

        val targs = listOf(c1, c4)

        val multiDs = Utils.toMultiDataSet(ds, listOf(fs1, fs2, fs3, fs4), targs, listOf(3, 3))

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
