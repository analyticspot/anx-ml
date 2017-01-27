package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.Column
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator
import org.slf4j.LoggerFactory

/**
 * To use DeepLearning4j's doc2vec implementation (which they call `ParagraphVectors`) you need a
 * `LabelAwareSentenceIterator`. Oddly, they use "sentence" when they mean document here
 * (see https://deeplearning4j.org/sentenceiterator). Their API does provide a `CollectionSentenceIterator` to allow you
 * to use a `List<String>` as a source of sentences but there isn't a similar tool for labelled sentences which is
 * what you'd want to do labelled doc2vec. This provides that functionality.
 *
 * Note that DeepLearning4j also has a `DocumentIterator` and a `LabelAwareDocumentIterator` but these treat a document
 * as an `InputStream` and allow only 1 label. There appears to be no difference in how the doc2vec code treats a
 * document and a sentence so we go with Sentence here. Indeed, a look at the source code for `ParagraphVectors` shows
 * that document and sentence iterators are treated identically.
 *
 * @param docColumn the column from which the documents will be pulled. The document is assumed to be already
 *     tokenized so it is represented as a `List<String>` -- a list of the tokens. It is up to the user if they want
 *     to include punctuation tokens to delimit sentences; the "right" answer will depend on the context. Tokens must
 *     not contain spaces.
 * @param labelColumns the columns from which the labels will be pulled.
 */
class TaggedDocumentIterator(val docColumn: Column<List<String>?>,
        val labelColumns: List<Column<String?>>)
    : LabelAwareSentenceIterator {

    // The row that will be used on the next call to nextSentence
    private var nextRow = 0
    // The current row is the one before the next row
    private val curRow: Int
        get() = nextRow - 1

    init {
        labelColumns.forEach {
            check(it.size == docColumn.size)
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(TaggedDocumentIterator::class.java)
    }

    override fun hasNext(): Boolean = nextRow < docColumn.size

    /**
     * Returns the next **document** (even though DeepLearning4j calls this a sentence).
     */
    override fun nextSentence(): String {
        check(hasNext())
        val tokens = docColumn[nextRow] ?: listOf()
        // Annoyingly DeepLearning4j ONLY supports un-tokenized strings and it insists on tokenize them itself so
        // here we concatenate the tokens with spaces and then use a simple space tokenizer to break them back apart.
        val result = tokens.joinToString(" ")
        ++nextRow
        return result
    }

    override fun currentLabels(): List<String> {
        check(curRow >= 0)
        return labelColumns.map { it[curRow] ?: "" }
    }

    override fun currentLabel(): String {
        val labels = currentLabels()
        check(labels.size == 1) {
            "currentLabel (single, as opposed to currentLabels) was called but ${labels.size} labels are available."
        }
        return labels[0]
    }

    override fun setPreProcessor(preProcessor: SentencePreProcessor?) {
        throw UnsupportedOperationException("not implemented")
    }

    override fun reset() {
        throw UnsupportedOperationException("not implemented")
    }

    override fun getPreProcessor(): SentencePreProcessor {
        throw UnsupportedOperationException("not implemented")
    }

    override fun finish() {
        // We don't anything to do in finish.
        log.debug("Finished called.")
    }
}
