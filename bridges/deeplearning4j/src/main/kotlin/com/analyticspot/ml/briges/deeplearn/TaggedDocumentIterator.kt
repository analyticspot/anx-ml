package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.Column
import org.deeplearning4j.text.documentiterator.LabelledDocument
import org.deeplearning4j.text.documentiterator.SimpleLabelAwareIterator
import org.slf4j.LoggerFactory

/**
 * To use DeepLearning4j's doc2vec implementation (which they call `ParagraphVectors`) you need a
 * `LabelAwareIterator`. Note that even though we've already tokenized the document DL4j doesn't support creating an
 * iterator in this way (confirmed on their chat with one of the devs): you **have to** concatenate all the tokens
 * into a single string and then, when you use it (e.g. with `ParagraphVectors`) supply a tokenizer that splits on
 * whitespace (the `DefaultTokenizer` does this).
 *
 * Note that DeepLearning4j also has a `DocumentIterator` and a `LabelAwareDocumentIterator` but these treat a document
 * as an `InputStream` and allow only 1 label.
 *
 * @param docColumn the column from which the documents will be pulled. The document is assumed to be already
 *     tokenized so it is represented as a `List<String>` -- a list of the tokens. It is up to the user if they want
 *     to include punctuation tokens to delimit sentences; the "right" answer will depend on the context. Tokens must
 *     not contain spaces.
 * @param labelColumns the columns from which the labels will be pulled.
 */
class TaggedDocumentIterator(val docColumn: Column<List<String>?>,
        val labelColumns: List<Column<String?>>)
    : SimpleLabelAwareIterator(toIterableOfLabelledDoc(docColumn, labelColumns)) {

    // The row that will be used on the next call to nextSentence
    private var nextRow = 0
    // The current row is the one before the next row
    private val curRow: Int
        get() = nextRow - 1

    companion object {
        private val log = LoggerFactory.getLogger(TaggedDocumentIterator::class.java)

        private fun toIterableOfLabelledDoc(
                docColumn: Column<List<String>?>, labelColumns: List<Column<String?>>): Iterable<LabelledDocument> {
            labelColumns.forEach {
                check(it.size == docColumn.size)
            }
            val labelledDocList = (0 until docColumn.size).map { idx ->
                val labels = labelColumns.mapNotNull { it[idx] }
                val doc = docColumn[idx]
                // As per class comment, we have to concatenate all the text for this to work
                val docStr = doc?.joinToString(" ") ?: ""
                DocWithLabels(docStr, labels)
            }
            return labelledDocList
        }
    }

    class DocWithLabels(content: String, labels: List<String>) : LabelledDocument() {
        init {
            this.content = content
            labels.forEach { this.addLabel(it) }
        }
    }
}
