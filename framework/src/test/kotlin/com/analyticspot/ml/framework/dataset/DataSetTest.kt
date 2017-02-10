package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.feature.CategoricalFeatureId
import com.analyticspot.ml.framework.feature.NumericalFeatureId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class DataSetTest {
    @Test
    fun testCanUseDataSetWithRegularColumnsAndFeatureColumns() {
        val c1Id = ColumnId.create<Int>("c1")
        val c2Id = CategoricalFeatureId("c2", false, setOf("a", "b"))
        val ds = DataSet.build {
            addColumn(c1Id, listOf(10, 20))
            addColumn(c2Id, listOf("a", "a"))
        }

        assertThat(ds.numColumns).isEqualTo(2)
        assertThat(ds.numRows).isEqualTo(2)

        // Create a FeatureDataSet and make sure we can merge, etc. with it and the regular data set
        val c3Id = NumericalFeatureId("c3", false)
        val fds = FeatureDataSet.build {
            addColumn(c3Id, listOf(18.0, 24.0))
        }

        val finalDs = DataSet.build {
            addAll(ds)
            addAll(fds)
        }

        assertThat(finalDs.numRows).isEqualTo(2)
        assertThat(finalDs.numColumns).isEqualTo(3)
        assertThat(finalDs.columnIds).containsExactly(c1Id, c2Id, c3Id)
    }
}
