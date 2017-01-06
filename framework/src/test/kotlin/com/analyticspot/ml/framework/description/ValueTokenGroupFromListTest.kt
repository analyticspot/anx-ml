package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.observation.SingleValueObservation
import com.analyticspot.ml.framework.testutils.WordCounts
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.util.concurrent.Executors

// This is really a test of several things: that ValueTokenGroup in general works, that the GraphExecutor can execute
// it, etc.
class ValueTokenGroupFromListTest {
    companion object {
        private val log = LoggerFactory.getLogger(ValueTokenGroupFromListTest::class.java)
    }

    @Test
    fun testBasicGraphWorks() {
        val srcId = ValueId.create<List<String>>("words")
        val wordGroupId = ValueIdGroup.create<Int>("wordCounts")
        val wordListId = ValueId.create<String>("wordList")
        val dg = DataGraph.build {
            val src = setSource {
                valueIds += srcId
            }

            // This is the transform that uses a ValueIdGroup/ValueTokenGroup.
            val wordCount = addTransform(src, WordCounts(src.token(srcId), wordGroupId))

            val tokenNames = addTransform(wordCount, TokenNamesTrans(wordCount.tokenGroup(wordGroupId), wordListId))

            val merge = merge(wordCount, tokenNames)

            result = merge
        }

        // Now run the transform and see what comes out the other side.
        val sourceSet = IterableDataSet(listOf(
                SingleValueObservation.create(listOf("foo", "bar", "bar")),
                SingleValueObservation.create(listOf("bar", "baz", "bar"))
        ))

        val resultDs = dg.trainTransform(sourceSet, Executors.newSingleThreadExecutor()).get().toList()
        assertThat(resultDs).hasSize(2)
        val firstRow = resultDs[0]
        // Note that in the following I'm relying on the fact that the words are assigned indices in the order that they
        // were encountered. Safe for the current implementation of the transform since that's just for testing.
        assertThat(firstRow.value(dg.result.token(wordListId))).isEqualTo("foo bar baz")
        assertThat(firstRow.values(dg.result.tokenGroup(wordGroupId))).isEqualTo(listOf(1, 2, 0))

        val secondRow = resultDs[1]
        assertThat(secondRow.value(dg.result.token(wordListId))).isEqualTo("foo bar baz")
        assertThat(secondRow.values(dg.result.tokenGroup(wordGroupId))).isEqualTo(listOf(0, 2, 1))
    }
}
