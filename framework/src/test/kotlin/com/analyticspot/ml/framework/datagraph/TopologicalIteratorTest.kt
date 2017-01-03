package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.ValueId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class TopologicalIteratorTest {
    @Test
    fun testIterationOfSingleNodeWorks() {
        val iter = TopologicalIterator(DataGraph.build {
            val source = setSource {
                valueIds += ValueId.create<String>("foo")
            }

            result = source
        })
        assertThat(iter.hasNext()).isTrue()
        val source = iter.next()
        assertThat(source).isInstanceOf(SourceGraphNode::class.java)
        assertThat(iter.hasNext()).isFalse()
    }
}
