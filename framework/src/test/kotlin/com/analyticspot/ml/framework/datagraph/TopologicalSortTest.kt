/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * Foobar is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.datagraph

import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.testutils.Graph1
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
                columnIds += ColumnId.create<String>("foo")
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
                columnIds += ColumnId.create<String>("foo")
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
            val sourceIds = listOf(ColumnId.create<Int>("v1"), ColumnId.create<Int>("v2"))
            val source = setSource {
                columnIds += sourceIds
            }

            val addC1 = addTransform(source, AddConstantTransform(11, source.transformDescription))

            val addC2 = addTransform(addC1, AddConstantTransform(12, addC1.transformDescription))

            val addC3 = addTransform(addC2, AddConstantTransform(88, addC2.transformDescription))

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
            val sourceIds = listOf(ColumnId.create<Int>("v1"), ColumnId.create<Int>("v2"))
            val source = setSource {
                columnIds += sourceIds
            }

            val addC1 = addTransform(source, AddConstantTransform(11, source.transformDescription))

            val addC2 = addTransform(source, AddConstantTransform(12, source.transformDescription))

            val addC3 = addTransform(source, AddConstantTransform(12, source.transformDescription))

            val merged = merge(addC1, addC2, addC3)

            val addC4 = addTransform(merged, AddConstantTransform(88, addC3.transformDescription))

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
            val sourceIds = listOf(ColumnId.create<Int>("v1"), ColumnId.create<Int>("v2"))
            val source = setSource {
                columnIds += sourceIds
            }

            val addC1 = addTransform(source, AddConstantTransform(11, source.transformDescription))

            val addC2 = addTransform(source, AddConstantTransform(12, source.transformDescription))

            val addC3 = addTransform(source, AddConstantTransform(12, source.transformDescription))

            // Two transforms in a pipeline from the output of c1
            val addC11 = addTransform(addC1, AddConstantTransform(19, addC1.transformDescription))

            val addC12 = addTransform(addC11, AddConstantTransform(19, addC11.transformDescription))

            // One transform from the output of c2
            val addC21 = addTransform(addC2, AddConstantTransform(37, addC2.transformDescription))

            // Now combine c3 and the results of the 2 other pipelines
            val merged = merge(addC3, addC12, addC21)

            result = merged
        }

        iterationIsOk(sort(dg).iterator(), setOf(), dg)
        iterationIsOk(sortWithTrain(dg).iterator(), setOf(), dg)
        backwardIterationIsOk(sortBackwards(dg).iterator(), setOf(), dg)
        backwardIterationIsOk(sortWithTrainBackwards(dg).iterator(), setOf(), dg)
    }

    @Test
    fun testComplexG1Works() {
        val g1 = Graph1()
        val dg = g1.graph

        val trainOnly = setOf(g1.invert1Node, g1.invert2Node)
        iterationIsOk(sort(dg).iterator(), trainOnly, dg)
        iterationIsOk(sortWithTrain(dg).iterator(), setOf(), dg)
        backwardIterationIsOk(sortBackwards(dg).iterator(), trainOnly, dg)
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
        assertThat(seenNodes.size + unexpectedNodes.size).isEqualTo(graph.allNodes.size)
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
        assertThat(seenNodes.size + unexpectedNodes.size).isEqualTo(graph.allNodes.size)
    }
}
