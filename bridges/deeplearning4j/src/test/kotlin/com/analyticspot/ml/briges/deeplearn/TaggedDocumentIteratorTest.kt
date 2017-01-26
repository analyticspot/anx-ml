package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class TaggedDocumentIteratorTest {
    @Test
    fun testNoLabels() {
        val docs = listOf(
                listOf("foo", "bar", "baz"),
                listOf("a", "b", "c", "d", "e", "f"),
                listOf("AnalyticSpot", "makes", "awesome", "software")
        )
        val docCol = ColumnId.create<List<String>>("doc")
        val ds = DataSet.build {
            addColumn(docCol, docs)
        }

        val iter = TaggedDocumentIterator(ds, docCol, listOf())

        assertThat(iter.hasNext()).isTrue()
        assertThat(iter.nextSentence()).isEqualTo(docs[0].joinToString(" "))
        assertThat(iter.currentLabels()).isEmpty()
        assertThat(iter.hasNext()).isTrue()
        assertThat(iter.nextSentence()).isEqualTo(docs[1].joinToString(" "))
        assertThat(iter.currentLabels()).isEmpty()
        assertThat(iter.hasNext()).isTrue()
        assertThat(iter.nextSentence()).isEqualTo(docs[2].joinToString(" "))
        assertThat(iter.currentLabels()).isEmpty()
        assertThat(iter.hasNext()).isFalse()

        iter.finish()
    }

    @Test
    fun testEmptyIterator() {
        val docCol = ColumnId.create<List<String>>("doc")
        val ds = DataSet.build {
            addColumn(docCol, listOf())
        }

        val iter = TaggedDocumentIterator(ds, docCol, listOf())

        assertThat(iter.hasNext()).isFalse()
        iter.finish()
    }

    @Test
    fun testWithOneLabel() {
        val docs = listOf(
                listOf("foo", "bar", "baz"),
                listOf("a", "b", "c", "d", "e", "f"),
                listOf("AnalyticSpot", "makes", "awesome", "software")
        )
        val docCol = ColumnId.create<List<String>>("doc")

        val labels = listOf("X", "Y", "Z")
        val labelCol = ColumnId.create<String>("label")
        val ds = DataSet.build {
            addColumn(docCol, docs)
            addColumn(labelCol, labels)
        }

        val iter = TaggedDocumentIterator(ds, docCol, listOf(labelCol))

        assertThat(iter.hasNext()).isTrue()
        assertThat(iter.nextSentence()).isEqualTo(docs[0].joinToString(" "))
        assertThat(iter.currentLabels()).containsExactly(labels[0])
        assertThat(iter.hasNext()).isTrue()
        assertThat(iter.nextSentence()).isEqualTo(docs[1].joinToString(" "))
        assertThat(iter.currentLabels()).containsExactly(labels[1])
        assertThat(iter.hasNext()).isTrue()
        assertThat(iter.nextSentence()).isEqualTo(docs[2].joinToString(" "))
        assertThat(iter.currentLabels()).containsExactly(labels[2])
        assertThat(iter.hasNext()).isFalse()

        iter.finish()
    }
}
