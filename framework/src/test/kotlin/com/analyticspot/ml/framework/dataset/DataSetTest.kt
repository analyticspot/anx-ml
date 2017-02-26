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

