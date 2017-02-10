package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.FeatureDataSet
import com.analyticspot.ml.framework.feature.CategoricalFeatureId
import com.analyticspot.ml.framework.feature.NumericalFeatureId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class DataConversionTest {
    @Test
    fun testDataSetConvertsCorrectly() {
        val catId1 = CategoricalFeatureId("c1", false, setOf("foo", "bar", "baz", "bizzle", "word"))
        val catId2 = CategoricalFeatureId("c2", true, setOf("x", "y", "z"))
        val numId1 = NumericalFeatureId("n1", false)
        val numId2 = NumericalFeatureId("n2", true)

        // note that we add columns here in alphabetical order by column id so make it easier to understand how the data
        // will be converted since the converted columns go in order of ds.columnIds where are alphabetical order.
        val ds = FeatureDataSet.build {
            addColumn(catId1, listOf("foo", "word"))
            addColumn(catId2, listOf("x", null))
            addColumn(numId1, listOf(1.0, 2.0))
            addColumn(numId2, listOf(null, 3.0))
        }

        val converted = DataConversion.fromDataSet(ds)

        assertThat(converted.attributes.map { it.name }).isEqualTo(ds.columnIds.map { it.name })
        assertThat(converted.data.size).isEqualTo(ds.numRows)

        // Check individual values. For numerical or missing we directly check the conversion. For non-missing
        // categorical we check that the corresponding attribute would correctly convert the string to the value and
        // vice-versa
        assertThat(converted.data[0].size).isEqualTo(ds.numColumns)
        val v00 = converted.data[0][0]
        assertThat(v00).isEqualTo(converted.attributes[0].valueOf("foo"))
        assertThat(converted.attributes[0].toString(v00)).isEqualTo("foo")

        val v01 = converted.data[0][1]
        assertThat(v01).isEqualTo(converted.attributes[1].valueOf("x"))
        assertThat(converted.attributes[1].toString(v01)).isEqualTo("x")

        assertThat(converted.data[0][2]).isEqualTo(1.0)

        assertThat(converted.data[0][3]).isNaN()

        // row 2
        assertThat(converted.data[1].size).isEqualTo(ds.numColumns)
        val v10 = converted.data[1][0]
        assertThat(v10).isEqualTo(converted.attributes[0].valueOf("word"))
        assertThat(converted.attributes[0].toString(v10)).isEqualTo("word")

        assertThat(converted.data[1][1]).isNaN()

        assertThat(converted.data[1][2]).isEqualTo(2.0)
        assertThat(converted.data[1][3]).isEqualTo(3.0)
    }
}
