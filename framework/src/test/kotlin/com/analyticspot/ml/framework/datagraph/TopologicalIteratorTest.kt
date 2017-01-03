package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.ValueId
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test

class TopologicalIteratorTest {
    companion object {
        private val log = LoggerFactory.getLogger(TopologicalIteratorTest::class.java)
    }

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

    @Test
    fun testGraph1Works() {
        val dg = DataGraph.build {
            val sourceIds = listOf(ValueId.create<Int>("v1"), ValueId.create<Int>("v2"))
            val source = setSource {
                valueIds += sourceIds
            }

            val c1ResId = ValueId.create<Int>("c1")
            val addC1 = addTransform(source, AddConstantTransform(11, source.token(sourceIds[0]), c1ResId))

            val c2ResId = ValueId.create<Int>("c2")
            val addC2 = addTransform(source, AddConstantTransform(12, source.token(sourceIds[0]), c2ResId))

            val c3ResId = ValueId.create<Int>("c3")
            val addC3 = addTransform(source, AddConstantTransform(12, source.token(sourceIds[0]), c3ResId))

            val merged = merge(addC1, addC2, addC3)

            val c4ResId = ValueId.create<Int>("c4")
            val addC4 = addTransform(merged, AddConstantTransform(88, addC3.token(c3ResId), c4ResId))

            result = addC4
        }

        iterationIsOk(TopologicalIterator(dg), dg)
    }

    // Ensures that the iteration is legal. Specifically, when each node is returned we've already seen all the nodes
    // that it uses as a source, the first node returned is the graph's source, and the last node returned is the
    // result. We also check in the other direction: all nodes listed as a node's subscribers are returned after
    // the current node. Other tests construct graphs and then just call this.
    private fun iterationIsOk(iter: Iterator<GraphNode>, graph: DataGraph) {
        val seenNodes = mutableSetOf<GraphNode>()
        if (!iter.hasNext()) {
            return
        }
        var curNode: GraphNode = iter.next()
        assertThat(curNode).isSameAs(graph.source)
        seenNodes.add(curNode)

        while (iter.hasNext()) {
            curNode = iter.next()
            log.debug("Iteration returned node {}. seenNodes: {}", curNode.id, seenNodes.map { it.id })
            curNode.sources.forEach {
                assertThat(seenNodes.contains(it)).isTrue()
            }
            curNode.subscribers.forEach {
                assertThat(seenNodes.contains(it)).isFalse()
            }
            seenNodes.add(curNode)
        }

        assertThat(curNode).isSameAs(graph.result)
    }
}
