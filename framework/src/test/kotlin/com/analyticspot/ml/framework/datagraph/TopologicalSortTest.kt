package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.ValueId
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.util.NoSuchElementException

// TODO: Add tests for train-only stuff once we add nodes/DataGraph ability to have such things.
class TopologicalSortTest {
    companion object {
        private val log = LoggerFactory.getLogger(TopologicalSortTest::class.java)
    }

    @Test
    fun testIterationOfSingleNodeWorks() {
        val iter = sort(DataGraph.build {
            val source = setSource {
                valueIds += ValueId.create<String>("foo")
            }

            result = source
        }).iterator()

        assertThat(iter.hasNext()).isTrue()
        val source = iter.next()
        assertThat(source).isInstanceOf(SourceGraphNode::class.java)
        assertThat(iter.hasNext()).isFalse()
    }

    @Test
    fun testEmptyThrowsCorrectException() {
        val iter = sort(DataGraph.build {
            val source = setSource {
                valueIds += ValueId.create<String>("foo")
            }

            result = source
        }).iterator()

        assertThat(iter.hasNext()).isTrue()
        val source = iter.next()
        assertThat(source).isInstanceOf(SourceGraphNode::class.java)
        assertThat(iter.hasNext()).isFalse()

        // As per spec of Iterator, this is the exception type that must be thrown.
        assertThatThrownBy { iter.next() }.isInstanceOf(NoSuchElementException::class.java)
    }

    @Test
    fun testSimplePipelineWorks() {
        val dg = DataGraph.build {
            val sourceIds = listOf(ValueId.create<Int>("v1"), ValueId.create<Int>("v2"))
            val source = setSource {
                valueIds += sourceIds
            }

            val c1ResId = ValueId.create<Int>("c1")
            val addC1 = addTransform(source, AddConstantTransform(11, source.token(sourceIds[0]), c1ResId))

            val c2ResId = ValueId.create<Int>("c2")
            val addC2 = addTransform(addC1, AddConstantTransform(12, addC1.token(c1ResId), c2ResId))

            val c3ResId = ValueId.create<Int>("c3")
            val addC3 = addTransform(addC2, AddConstantTransform(88, addC2.token(c2ResId), c3ResId))

            result = addC3
        }

        iterationIsOk(sort(dg).iterator(), setOf(), dg)
        iterationIsOk(sortWithTrain(dg).iterator(), setOf(), dg)
        backwardIterationIsOk(sortBackwards(dg).iterator(), setOf(), dg)
        backwardIterationIsOk(sortWithTrainBackwards(dg).iterator(), setOf(), dg)
    }

    // This tests a graph where the source feeds into 3 AddConstantTransforms, those are then merged into a single
    // data set, and then then feeds into yet another AddConstantTransform which is also the final output.
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
            val addC2 = addTransform(source, AddConstantTransform(12, source.token(sourceIds[1]), c2ResId))

            val c3ResId = ValueId.create<Int>("c3")
            val addC3 = addTransform(source, AddConstantTransform(12, source.token(sourceIds[0]), c3ResId))

            val merged = merge(addC1, addC2, addC3)

            val c4ResId = ValueId.create<Int>("c4")
            val addC4 = addTransform(merged, AddConstantTransform(88, addC3.token(c3ResId), c4ResId))

            result = addC4
        }

        iterationIsOk(sort(dg).iterator(), setOf(), dg)
        iterationIsOk(sortWithTrain(dg).iterator(), setOf(), dg)
        backwardIterationIsOk(sortBackwards(dg).iterator(), setOf(), dg)
        backwardIterationIsOk(sortWithTrainBackwards(dg).iterator(), setOf(), dg)
    }

    // Tests a graph with "unequal legs". The source feeds into 3 transforms, C1, C2, and C3. C1 then goes through a
    // pipeline of 2 more transforms: c11 and c12, c2 goes through one more transform c21. Then the outputs of
    // c12, c21 and c3 are fed into a merge. The merge is the final result.
    @Test
    fun testGraph2Works() {
        val dg = DataGraph.build {
            val sourceIds = listOf(ValueId.create<Int>("v1"), ValueId.create<Int>("v2"))
            val source = setSource {
                valueIds += sourceIds
            }

            val c1ResId = ValueId.create<Int>("c1")
            val addC1 = addTransform(source, AddConstantTransform(11, source.token(sourceIds[0]), c1ResId))

            val c2ResId = ValueId.create<Int>("c2")
            val addC2 = addTransform(source, AddConstantTransform(12, source.token(sourceIds[1]), c2ResId))

            val c3ResId = ValueId.create<Int>("c3")
            val addC3 = addTransform(source, AddConstantTransform(12, source.token(sourceIds[0]), c3ResId))

            // Two transforms in a pipeline from the output of c1
            val c11ResId = ValueId.create<Int>("c1.1")
            val addC11 = addTransform(addC1, AddConstantTransform(19, addC1.token(c1ResId), c11ResId))

            val c12ResId = ValueId.create<Int>("c1.2")
            val addC12 = addTransform(addC11, AddConstantTransform(19, addC11.token(c11ResId), c12ResId))

            // One transform from the output of c2
            val c21ResId = ValueId.create<Int>("c2.1")
            val addC21 = addTransform(addC2, AddConstantTransform(37, addC2.token(c2ResId), c21ResId))

            // Now combine c3 and the results of the 2 other pipelines
            val merged = merge(addC3, addC12, addC21)

            result = merged
        }

        iterationIsOk(sort(dg).iterator(), setOf(), dg)
        iterationIsOk(sortWithTrain(dg).iterator(), setOf(), dg)
        backwardIterationIsOk(sortBackwards(dg).iterator(), setOf(), dg)
        backwardIterationIsOk(sortWithTrainBackwards(dg).iterator(), setOf(), dg)
    }

    // Ensures that the iteration is legal. Specifically, when each node is returned we've already seen all the nodes
    // that it uses as a source, the first node returned is the graph's source, and the last node returned is the
    // result. We also check in the other direction: all nodes listed as a node's subscribers are returned after
    // the current node. Other tests construct graphs and then just call this. This also checks that we don't see
    // anything in unexpectedNodes (generally these are train-only nodes with non-training iterators) and that we see
    // all nodes in the graph except those in unexpectedNodes.
    private fun iterationIsOk(iter: Iterator<GraphNode>, unexpectedNodes: Set<GraphNode>, graph: DataGraph) {
        val seenNodes = mutableSetOf<GraphNode>()
        if (!iter.hasNext()) {
            return
        }
        var curNode: GraphNode = iter.next()
        assertThat(curNode).isSameAs(graph.source)
        assertThat(unexpectedNodes).doesNotContain(curNode)
        seenNodes.add(curNode)

        while (iter.hasNext()) {
            curNode = iter.next()
            log.debug("Iteration returned node {}. seenNodes: {}", curNode.id, seenNodes.map { it.id })
            assertThat(unexpectedNodes).doesNotContain(curNode)
            curNode.sources.forEach {
                assertThat(seenNodes.contains(it.source)).isTrue()
            }
            curNode.subscribers.forEach {
                assertThat(seenNodes.contains(it.subscriber)).isFalse()
            }
            seenNodes.add(curNode)
        }

        assertThat(curNode).isSameAs(graph.result)
        assertThat(seenNodes.size - unexpectedNodes.size).isEqualTo(graph.allNodes.size)
    }

    // Ensures that the backward iteration is legal. see iteratorIsOK.
    private fun backwardIterationIsOk(iter: Iterator<GraphNode>, unexpectedNodes: Set<GraphNode>, graph: DataGraph) {
        val seenNodes = mutableSetOf<GraphNode>()
        if (!iter.hasNext()) {
            return
        }
        var curNode: GraphNode = iter.next()
        assertThat(curNode).isSameAs(graph.result)
        assertThat(unexpectedNodes).doesNotContain(curNode)
        seenNodes.add(curNode)

        while (iter.hasNext()) {
            curNode = iter.next()
            log.debug("Iteration returned node {}. seenNodes: {}", curNode.id, seenNodes.map { it.id })
            assertThat(unexpectedNodes).doesNotContain(curNode)
            curNode.sources.forEach {
                assertThat(seenNodes.contains(it.source)).isFalse()
            }
            curNode.subscribers.forEach {
                assertThat(seenNodes.contains(it.subscriber)).isTrue()
            }
            seenNodes.add(curNode)
        }

        assertThat(curNode).isSameAs(graph.source)
        assertThat(seenNodes.size - unexpectedNodes.size).isEqualTo(graph.allNodes.size)
    }
}
