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
        assertThat(iter.hasNext()).isTrue()
        assertThat(iter.nextSentence()).isEqualTo(docs[1].joinToString(" "))
        assertThat(iter.hasNext()).isTrue()
        assertThat(iter.nextSentence()).isEqualTo(docs[2].joinToString(" "))
        assertThat(iter.hasNext()).isFalse()
    }
}
