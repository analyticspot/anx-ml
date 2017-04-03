package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class UtilsTest {
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
        val fs1 = listOf(c1, c2)
        val fs2 = listOf(c2)
        val fs3 = listOf(c2, c4)
        val fs4 = listOf(c2, c3, c4)

        val targs = listOf(c1, c4)

        val multiDs = Utils.toMultiDataSet(ds, listOf(fs1, fs2, fs3, fs4), targs, listOf(3, 3))

        assertThat(multiDs.features.size).isEqualTo(4)
        assertThat(multiDs.features[0].shape()).isEqualTo(arrayOf(3, 2))
        assertThat(multiDs.features[1].shape()).isEqualTo(arrayOf(3, 1))
        assertThat(multiDs.features[2].shape()).isEqualTo(arrayOf(3, 2))
        assertThat(multiDs.features[3].shape()).isEqualTo(arrayOf(3, 3))


    }
}
