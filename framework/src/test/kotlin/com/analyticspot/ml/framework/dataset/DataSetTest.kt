package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.ColumnIdGroup
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import com.analyticspot.ml.framework.metadata.MaybeMissingMetaData
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.io.ByteArrayOutputStream

class DataSetTest {
    companion object {
        private val log = LoggerFactory.getLogger(DataSetTest::class.java)
    }

    @Test
    fun testToDelimited() {
        val col1 = ColumnId.create<Int>("c1")
        val col2 = ColumnId.create<String>("c2")
        val col3 = ColumnId.create<Double>("c3")
        val ds = DataSet.build {
            addColumn(col1, listOf(1, 2, null))
            addColumn(col2, listOf("foo", null, "bar"))
            addColumn(col3, listOf(null, 1.002, 0.0004))
        }

        val output = ByteArrayOutputStream()
        val nullVal = "NULL"
        val delimiter = ","
        ds.toDelimited(output, true, nullVal, delimiter)
        output.close()

        val outputStr = String(output.toByteArray())

        val expected = """
        |c1,c2,c3
        |1,foo,NULL
        |2,NULL,1.002
        |NULL,bar,4.0E-4
        |""".trimMargin()

        assertThat(outputStr).isEqualTo(expected)
    }

    @Test
    fun testToDelimitedDoesNotCloseStream() {
        val col1 = ColumnId.create<Int>("c1")
        val col2 = ColumnId.create<String>("c2")
        val col3 = ColumnId.create<Double>("c3")
        val ds = DataSet.build {
            addColumn(col1, listOf(1, 2, null))
            addColumn(col2, listOf("foo", null, "bar"))
            addColumn(col3, listOf(null, 1.002, 0.0004))
        }

        val output = ByteArrayOutputStream()
        ds.toDelimited(output)

        // Now write some more stuff to the stream to ensure it wasn't closed
        val finalLine = "AND MORE STUFF"
        val writer = output.writer()
        writer.write("\n$finalLine")
        writer.close()

        val outputStr = String(output.toByteArray())
        assertThat(outputStr).endsWith(finalLine)
    }

    @Test
    fun testColumnSubsets() {
        val col1 = ColumnId.create<Int>("c1")
        val metaData1 = MaybeMissingMetaData(true)
        val col2 = ColumnId.create<String>("c2")
        val metaData2 = CategoricalFeatureMetaData(true, setOf("foo", "bar"))
        val col3 = ColumnId.create<Double>("c3")
        val ds = DataSet.build {
            addColumn(col1, listOf(1, 2, null), metaData1)
            addColumn(col2, listOf("foo", null, "bar"), metaData2)
            addColumn(col3, listOf(17.0, 1.002, 0.0004))
        }

        val ds2 = ds.columnSubset(col1, col3)
        assertThat(ds2.numColumns).isEqualTo(2)
        assertThat(ds2.columnIds).containsExactly(col1, col3)
        assertThat(ds2.metaData["c1"]).isEqualTo(metaData1)
        assertThat(ds2.metaData["c3"]).isNull()
        assertThat(ds2.column(col1)).containsExactlyElementsOf(ds.column(col1))
        assertThat(ds2.column(col3)).containsExactlyElementsOf(ds.column(col3))

        // Similar to the above, but pass a list of columns instead
        val ds3 = ds.columnSubset(listOf(col1, col3))
        assertThat(ds3.numColumns).isEqualTo(2)
        assertThat(ds3.columnIds).containsExactly(col1, col3)
        assertThat(ds3.metaData["c1"]).isEqualTo(metaData1)
        assertThat(ds3.metaData["c3"]).isNull()
        assertThat(ds3.column(col1)).containsExactlyElementsOf(ds.column(col1))
        assertThat(ds3.column(col3)).containsExactlyElementsOf(ds.column(col3))

        // Try the overload where we pass just the column names
        val ds4 = ds.columnSubset("c1", "c3")
        assertThat(ds4.numColumns).isEqualTo(2)
        assertThat(ds4.columnIds).containsExactly(col1, col3)
        assertThat(ds4.metaData["c1"]).isEqualTo(metaData1)
        assertThat(ds4.metaData["c3"]).isNull()
        assertThat(ds4.column(col1)).containsExactlyElementsOf(ds.column(col1))
        assertThat(ds4.column(col3)).containsExactlyElementsOf(ds.column(col3))
    }

    @Test
    fun testRowSubset() {
        val cid1 = ColumnId.create<Int>("c1")
        val cid2 = ColumnId.create<String>("c2")

        val ds = DataSet.build {
            addColumn(cid1, listOf(1, 2, 3, 4, 5), MaybeMissingMetaData(false))
            addColumn(cid2, listOf("a", "b", "c", "d", "e"), MaybeMissingMetaData(true))
        }

        val rowsToKeep = setOf(0, 3, 4)
        val rowSub = ds.rowsSubset(rowsToKeep)
        assertThat(rowSub.numColumns).isEqualTo(2)
        assertThat(rowSub.numRows).isEqualTo(rowsToKeep.size)
        assertThat(rowSub.metaData[cid1.name]).isEqualTo(ds.metaData[cid1.name])
        assertThat(rowSub.metaData[cid2.name]).isEqualTo(ds.metaData[cid2.name])

        assertThat(rowSub.value(0, cid1)).isEqualTo(1)
        assertThat(rowSub.value(1, cid1)).isEqualTo(4)
        assertThat(rowSub.value(2, cid1)).isEqualTo(5)

        assertThat(rowSub.value(0, cid2)).isEqualTo("a")
        assertThat(rowSub.value(1, cid2)).isEqualTo("d")
        assertThat(rowSub.value(2, cid2)).isEqualTo("e")
    }

    @Test
    fun testRandomSubsets() {
        val cid1 = ColumnId.create<Int>("c1")
        val cid2 = ColumnId.create<String>("c2")

        val ds = DataSet.build {
            addColumn(cid1, listOf(1, 2, 3, 4, 5), MaybeMissingMetaData(false))
            addColumn(cid2, listOf("a", "b", "c", "d", "e"), MaybeMissingMetaData(true))
        }

        val (sub1, sub2) = ds.randomSubsets(2)
        assertThat(sub1.numRows).isEqualTo(2)
        assertThat(sub2.numRows).isEqualTo(3)
        assertThat(sub1.column(cid1)).doesNotContainAnyElementsOf(sub2.column(cid1))
        assertThat(sub1.column(cid2)).doesNotContainAnyElementsOf(sub2.column(cid2))
    }

    @Test
    fun testColumnIdsInGroupReturnsCorrectColumns() {
        val col1 = ColumnId.create<String>("aaa")
        val col2 = ColumnId.create<String>("bbb")
        // Should be OK that this has the name prefix as col2
        val colGroup = ColumnIdGroup.create<String>("bbb")
        val col3 = ColumnId.create<String>("ccc")
        val col4 = ColumnId.create<String>("ddd")

        val suffixesInGroup = listOf("foo", "bar", "baz", "bip", "bumble")

        val ds = DataSet.build {
            addColumn(col1, listOf("foo"))
            addColumn(col2, listOf("foo"))
            addColumn(col3, listOf("foo"))
            addColumn(col4, listOf("foo"))

            suffixesInGroup.forEach {
                addColumn(colGroup.generateId(it), listOf("hello"))
            }
        }

        val colsInGroup = ds.colIdsInGroup(colGroup)

        assertThat(colsInGroup.toSet()).isEqualTo(suffixesInGroup.map { colGroup.generateId(it) }.toSet())
    }

    @Test
    fun testColumnIdsInGroupOkWithEmptyGroup() {
        val col1 = ColumnId.create<String>("aaa")
        val col2 = ColumnId.create<String>("bbb")
        val col3 = ColumnId.create<String>("ccc")
        val col4 = ColumnId.create<String>("ddd")

        val colGroup = ColumnIdGroup.create<String>("bbb")

        val ds = DataSet.build {
            addColumn(col1, listOf("foo"))
            addColumn(col2, listOf("foo"))
            addColumn(col3, listOf("foo"))
            addColumn(col4, listOf("foo"))
        }

        val colsInGroup = ds.colIdsInGroup(colGroup)

        assertThat(colsInGroup).isEmpty()
    }

    @Test
    fun testColumnIdsInGroupOkWithOnlyGroup() {
        val colGroup = ColumnIdGroup.create<String>("group")

        val suffixesInGroup = listOf("foo", "bar", "baz", "bip", "bumble")
        val ds = DataSet.build {
            suffixesInGroup.forEach {
                addColumn(colGroup.generateId(it), listOf("hello"))
            }
        }

        val colsInGroup = ds.colIdsInGroup(colGroup)
        assertThat(colsInGroup.toSet()).isEqualTo(suffixesInGroup.map { colGroup.generateId(it) }.toSet())
    }

    @Test
    fun testCanSerializeAndDeserialize() {
        val ds = DataSet.build {
            addColumn(ColumnId.create<String>("c1"), listOf("a", "b", "c"),
                    CategoricalFeatureMetaData(false, setOf("a", "b", "c", "d")))
            addColumn(ColumnId.create<Int>("c2"), listOf(4, 5, 6))
            addColumn(ColumnId.create<Int>("c3"), listOf(7, null, 8), MaybeMissingMetaData(true))
        }

        val serialized = ds.saveToString()
        log.debug("DataSet serialized as: {}", serialized)

        val deserDs = DataSet.fromSaved(serialized)

        assertThat(deserDs.numColumns).isEqualTo(ds.numColumns)
        assertThat(deserDs.numRows).isEqualTo(ds.numRows)
        assertThat(deserDs.columnIds).isEqualTo(ds.columnIds)
        for (cid in deserDs.columnIds) {
            assertThat(deserDs.column(cid).toList()).isEqualTo(ds.column(cid).toList())
        }

    }
}

