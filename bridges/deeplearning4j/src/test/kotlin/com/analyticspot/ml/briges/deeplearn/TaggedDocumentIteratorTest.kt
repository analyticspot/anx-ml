package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.testng.annotations.BeforeClass
import org.testng.annotations.Test

class TaggedDocumentIteratorTest {

    @BeforeClass
    fun globalSetup() {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    }

    @Test
    fun testNoLabels() {
        val docs = listOf(
                "foo bar .",
                "a b c d e f",
                "AnalyticSpot makes awesome software !"
        )
        val docCol = ColumnId.create<String>("doc")
        val ds = DataSet.build {
            addColumn(docCol, docs)
        }

        val iter = TaggedDocumentIterator(ds.column(docCol), listOf())

        assertThat(iter.hasNext()).isTrue()
        var doc = iter.nextDocument()
        assertThat(doc.content).isEqualTo(docs[0])
        assertThat(doc.labels).isEmpty()

        assertThat(iter.hasNext()).isTrue()
        doc = iter.nextDocument()
        assertThat(doc.content).isEqualTo(docs[1])
        assertThat(doc.labels).isEmpty()

        assertThat(iter.hasNext()).isTrue()
        doc = iter.nextDocument()
        assertThat(doc.content).isEqualTo(docs[2])
        assertThat(doc.labels).isEmpty()

        assertThat(iter.hasNext()).isFalse()
    }

    @Test
    fun testEmptyIterator() {
        val docCol = ColumnId.create<String>("doc")
        val ds = DataSet.build {
            addColumn(docCol, listOf())
        }

        val iter = TaggedDocumentIterator(ds.column(docCol), listOf())

        assertThat(iter.hasNext()).isFalse()
    }

    @Test
    fun testWithOneLabel() {
        val docs = listOf(
                "foo ! baz",
                "a b c d e f",
                "AnalyticSpot makes awesome software !! Really ."
        )
        val docCol = ColumnId.create<String>("doc")

        val labels = listOf("X", "Y", "Z")
        val labelCol = ColumnId.create<String>("label")
        val ds = DataSet.build {
            addColumn(docCol, docs)
            addColumn(labelCol, labels)
        }

        val iter = TaggedDocumentIterator(ds.column(docCol), listOf(ds.column(labelCol)))

        assertThat(iter.hasNext()).isTrue()
        var doc = iter.nextDocument()
        assertThat(doc.content).isEqualTo(docs[0])
        assertThat(doc.labels).containsExactly(labels[0])

        assertThat(iter.hasNext()).isTrue()
        doc = iter.nextDocument()
        assertThat(doc.content).isEqualTo(docs[1])
        assertThat(doc.labels).containsExactly(labels[1])

        assertThat(iter.hasNext()).isTrue()
        doc = iter.nextDocument()
        assertThat(doc.content).isEqualTo(docs[2])
        assertThat(doc.labels).containsExactly(labels[2])

        assertThat(iter.hasNext()).isFalse()
    }

    @Test
    fun testWithManyLabels() {
        val docs = listOf(
                "foo ! baz",
                "a b c d e f",
                "AnalyticSpot makes awesome software !! Really ."
        )
        val docCol = ColumnId.create<String>("doc")

        val labels1 = listOf("X", "Y", "Z")
        val label1Col = ColumnId.create<String>("label1")

        val labels2 = listOf("be", "bop", "bazzle")
        val label2Col = ColumnId.create<String>("label2")

        val labels3 = listOf("bi", "bim", "bap")
        val label3Col = ColumnId.create<String>("label3")

        val ds = DataSet.build {
            addColumn(docCol, docs)
            addColumn(label1Col, labels1)
            addColumn(label2Col, labels2)
            addColumn(label3Col, labels3)
        }

        val iter = TaggedDocumentIterator(ds.column(docCol),
                listOf(ds.column(label1Col), ds.column(label2Col), ds.column(label3Col)))

        assertThat(iter.hasNext()).isTrue()
        var doc = iter.nextDocument()
        assertThat(doc.content).isEqualTo(docs[0])
        assertThat(doc.labels).containsExactly(labels1[0], labels2[0], labels3[0])

        assertThat(iter.hasNext()).isTrue()
        doc = iter.nextDocument()
        assertThat(doc.content).isEqualTo(docs[1])
        assertThat(doc.labels).containsExactly(labels1[1], labels2[1], labels3[1])

        assertThat(iter.hasNext()).isTrue()
        doc = iter.nextDocument()
        assertThat(doc.content).isEqualTo(docs[2])
        assertThat(doc.labels).containsExactly(labels1[2], labels2[2], labels3[2])

        assertThat(iter.hasNext()).isFalse()
    }
}
