package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import com.analyticspot.ml.framework.metadata.MaybeMissingMetaData
import com.analyticspot.ml.framework.serialization.JsonMapper
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test
import java.util.concurrent.Executors

class AllColumnsExceptTransformTest {
    @Test
    fun testWorks() {
        val c1Id = ColumnId.create<String>("foo")
        val c1Meta = CategoricalFeatureMetaData(false, setOf("a", "b", "c"))

        val c2Id = ColumnId.create<Int>("bar")
        val c2Meta = MaybeMissingMetaData(false)

        val c3Id = ColumnId.create<Double>("baz")

        val c4Id = ColumnId.create<Double>("bop")
        val c4Meta = MaybeMissingMetaData(true)

        val ds = DataSet.build {
            addColumn(c1Id, listOf("a"), c1Meta)
            addColumn(c2Id, listOf(1), c2Meta)
            addColumn(c3Id, listOf(10.0))
            addColumn(c4Id, listOf(18.0), c4Meta)
        }

        val subT1 = AllColumnsExceptTransform.create("baz")

        val exec = Executors.newSingleThreadExecutor()
        val subT1Result = subT1.transform(ds, exec).get()

        assertThat(subT1Result.numRows).isEqualTo(1)
        assertThat(subT1Result.numColumns).isEqualTo(3)
        assertThat(subT1Result.columnIds).contains(c1Id, c2Id, c4Id)
        assertThat(subT1Result.columnIds).containsOnly(c1Id, c2Id, c4Id)
        assertThat(subT1Result.metaData).isEqualTo(mapOf(c1Id.name to c1Meta, c2Id.name to c2Meta, c4Id.name to c4Meta))

        val subT2 = AllColumnsExceptTransform.create("bar", "bop")

        val subT2Result = subT2.transform(ds, exec).get()

        assertThat(subT2Result.numRows).isEqualTo(1)
        assertThat(subT2Result.numColumns).isEqualTo(2)
        assertThat(subT2Result.columnIds).contains(c1Id, c3Id)
        assertThat(subT2Result.columnIds).containsOnly(c1Id, c3Id)
        assertThat(subT2Result.metaData).isEqualTo(mapOf(c1Id.name to c1Meta))
    }

    @Test
    fun testCanSerDeser() {
        val subT1 = AllColumnsExceptTransform.create("baz")

        val serialized = JsonMapper.mapper.writeValueAsString(subT1)

        val deser = JsonMapper.mapper.readValue(serialized, AllColumnsExceptTransform::class.java)

        assertThat(deser.toExclude).isEqualTo(setOf("baz"))
    }
}
