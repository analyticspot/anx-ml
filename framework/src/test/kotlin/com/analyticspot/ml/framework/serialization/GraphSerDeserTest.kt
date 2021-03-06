/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * The ANX ML library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.AddConstantTransform
import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.datagraph.DataSetSourceGraphNode
import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datagraph.SourceGraphNode
import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.datatransform.MergeTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.ColumnIdGroup
import com.analyticspot.ml.framework.testutils.Graph1
import com.analyticspot.ml.framework.testutils.LowerCaseTransform
import com.analyticspot.ml.framework.testutils.WordCounts
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.concurrent.Executors

class GraphSerDeserTest {
    companion object {
        private val log = LoggerFactory.getLogger(GraphSerDeserTest::class.java)
    }

    // Serialize a simple AddConstantTransform and then deserialize it with the factory. Changes the source GraphNode so
    // that we can be sure that the source tokens are regenerated.
    @Test
    fun testCanSerializeAndDeserializeSimpleTransform() {
        // Create a source with two valueIds, the 2nd is in the input to our transform
        val valIdToTransform = ColumnId.create<Int>("val2")

        // Construct the transform
        val amountToAdd = 11
        val trans = AddConstantTransform(amountToAdd)

        val serDeser = GraphSerDeser()

        // Serialize it to the output stream
        val output = ByteArrayOutputStream()
        serDeser.serializeTransform(trans, output)
        log.debug("Transform serialized as: {}", output.toString())

        val input = ByteArrayInputStream(output.toByteArray())

        // Construct a new source with only 1 token. Note that the index of this token will be different than the one
        // from the original source.
        val newSource = SourceGraphNode.build(0) {
            columnIds += valIdToTransform
        }
        // Now deserialize relative to this new source
        val deserialized = serDeser.deserializeTransform(
                null, StandardJsonFormat.MetaData(trans), listOf(newSource), input)

        assertThat(deserialized).isInstanceOf(AddConstantTransform::class.java)
        val deserializedAddConstant = deserialized as AddConstantTransform
        assertThat(deserializedAddConstant.toAdd).isEqualTo(amountToAdd)
    }

    @Test
    fun testCanSerializeAndDeserializeMerge() {
        val v1Id = ColumnId.create<String>("v1")
        // Note: Not using sequential numbers here to ensure things deserialize using the correct data sets.
        val s1 = SourceGraphNode.build(12) {
            columnIds += v1Id
        }

        val v2Id = ColumnId.create<String>("v2")
        val s2 = SourceGraphNode.build(8) {
            columnIds += v2Id
        }

        val merge = MergeTransform.build {
            sources += listOf(s1, s2)
        }

        val serDeser = GraphSerDeser()
        val output = ByteArrayOutputStream()
        serDeser.serializeTransform(merge, output)

        log.debug("Merge transform serializaed as: {}", output.toString())

        val input = ByteArrayInputStream(output.toByteArray())

        val deserialized = serDeser.deserializeTransform(
                null, StandardJsonFormat.MetaData(merge), listOf(s1, s2), input)

        assertThat(deserialized).isInstanceOf(MergeTransform::class.java)
    }

    @Test
    fun testCanSerializeAndDeserializeSimpleGraph() {
        val sourceColId = ColumnId.create<Int>("src")
        val amountToAdd = 1232
        val dg = DataGraph.build {
            val source = setSource {
                columnIds += sourceColId
            }

            val trans = addTransform(
                    source, AddConstantTransform(amountToAdd))
            result = trans
        }

        val serDeser = GraphSerDeser()
        val outStream = ByteArrayOutputStream(0)
        serDeser.serialize(dg, outStream)

        // Now deserialize the thing....
        val inStream = ByteArrayInputStream(outStream.toByteArray())
        val deserGraph = serDeser.deserialize(inStream)

        assertThat(deserGraph.source).isInstanceOf(SourceGraphNode::class.java)
        assertThat(deserGraph.metaData).isNull()
        val dss = deserGraph.source as SourceGraphNode
        assertThat(dss.columnIds).hasSize(1)
        assertThat(dss.columnIds[0]).isEqualTo(sourceColId)

        val sourceValue = 18
        val sourceData = deserGraph.createSource(sourceValue)
        val result = deserGraph.transform(sourceData, Executors.newSingleThreadExecutor()).get()

        assertThat(result.value(0, sourceColId)).isEqualTo(sourceValue + amountToAdd)
    }

    @Test
    fun testMetaDataIsSerialized() {
        val sourceColId = ColumnId.create<Int>("src")
        val amountToAdd = 1232
        val metaData = "some random metadata"
        val dg = DataGraph.build {
            val source = setSource {
                columnIds += sourceColId
            }

            val trans = addTransform(
                    source, AddConstantTransform(amountToAdd))
            result = trans
        }
        dg.metaData = metaData

        val serDeser = GraphSerDeser()
        val outStream = ByteArrayOutputStream(0)
        serDeser.serialize(dg, outStream)

        // Now deserialize the thing....
        val inStream = ByteArrayInputStream(outStream.toByteArray())
        val deserGraph = serDeser.deserialize(inStream)

        assertThat(deserGraph.metaData).isEqualTo(metaData)
    }

    @Test
    fun testCanSerAndDeserColumnSubset() {
        val c1 = ColumnId.create<String>("c1")
        val c2 = ColumnId.create<String>("c2")
        val c3 = ColumnId.create<String>("c3")

        val dg = DataGraph.build {
            val src = setSource {
                columnIds += listOf(c1, c2, c3)
            }

            val sub = subsetColumns(src) {
                keep(c1)
                keepAndRename(c3, "renamed")
            }

            result = sub
        }

        val serDeser = GraphSerDeser()
        val outStream = ByteArrayOutputStream()
        serDeser.serialize(dg, outStream)

        val deser = serDeser.deserialize(ByteArrayInputStream(outStream.toByteArray()))

        val testDs = dg.createSource("foo", "bar", "baz")

        val result = deser.transform(testDs, Executors.newSingleThreadExecutor()).get()

        assertThat(result.columnIds).containsExactly(c1, ColumnId.create<String>("renamed"))
    }

    @Test
    fun testCanSerializeAndDeserializeDataSetSource() {
        val sourceColId = ColumnId.create<Int>("src")
        val amountToAdd = 1232
        val dg = DataGraph.build {
            val source = dataSetSource()

            val trans = addTransform(source, AddConstantTransform(amountToAdd))
            result = trans
        }

        val serDeser = GraphSerDeser()
        val outStream = ByteArrayOutputStream(0)
        serDeser.serialize(dg, outStream)

        // Now deserialize the thing....
        val inStream = ByteArrayInputStream(outStream.toByteArray())
        val deserGraph = serDeser.deserialize(inStream)

        assertThat(deserGraph.source).isInstanceOf(DataSetSourceGraphNode::class.java)

        val sourceValue = 18
        val sourceData = DataSet.create(sourceColId, listOf(sourceValue))
        val result = deserGraph.transform(sourceData, Executors.newSingleThreadExecutor()).get()

        assertThat(result.value(0, sourceColId)).isEqualTo(sourceValue + amountToAdd)
    }

    @Test
    fun testSourceGraphNodeSerializesTrainOnlyInformation() {
        val sourceColIds = listOf(
                ColumnId.create<Int>("src1"),
                ColumnId.create<String>("src2"))

        val trainOnlySourceColIds = listOf(
                ColumnId.create<Boolean>("srcT1"),
                ColumnId.create<Int>("srcT2"))

        val dg = DataGraph.build {
            val source = setSource {
                columnIds += sourceColIds
                trainOnlyColumnIds += trainOnlySourceColIds
            }

            result = source
        }

        val serDeser = GraphSerDeser()
        val outStream = ByteArrayOutputStream()
        serDeser.serialize(dg, outStream)

        // Now deserialize the thing....
        val inStream = ByteArrayInputStream(outStream.toByteArray())
        val deserGraph = serDeser.deserialize(inStream)

        assertThat(deserGraph.source).isInstanceOf(SourceGraphNode::class.java)
        val dss = deserGraph.source as SourceGraphNode
        assertThat(dss.columnIds).isEqualTo(sourceColIds.plus(trainOnlySourceColIds))
        assertThat(dss.trainOnlyColumnIds).isEqualTo(trainOnlySourceColIds)
    }

    @Test
    fun testColumnIdGroupsSerialize() {
        val srcId = ColumnId.create<List<String>>("words")
        val wordGroupId = ColumnIdGroup.create<Int>("wordCounts")
        val dg = DataGraph.build {
            val src = setSource {
                columnIds += srcId
            }

            // This is the transform that uses a ColumnIdGroup
            val wordCount = addTransform(src, WordCounts(srcId, wordGroupId))

            result = wordCount
        }

        // Now run the transform and see what comes out the other side.
        val sourceSet = dg.createSource(listOf(
                listOf(listOf("foo", "bar", "bar")),
                listOf(listOf("bar", "baz", "bar"))
        ))

        // Train it.
        dg.trainTransform(sourceSet, Executors.newSingleThreadExecutor()).get()

        // Now serialize it.
        val serDeser = GraphSerDeser()
        val output = ByteArrayOutputStream()
        serDeser.serialize(dg, output)

        // And deserilize it
        val deserDg = serDeser.deserialize(ByteArrayInputStream(output.toByteArray()))

        val toTransform = dg.createSource(listOf("foo", "bar", "foo", "foo", "frabble"))

        val resultDs = deserDg.transform(toTransform, Executors.newSingleThreadExecutor()).get()

        assertThat(resultDs.numRows).isEqualTo(1)
        assertThat(resultDs.numColumns).isEqualTo(3)
        assertThat(resultDs.value(0, wordGroupId.generateId("foo"))).isEqualTo(3)
        assertThat(resultDs.value(0, wordGroupId.generateId("bar"))).isEqualTo(1)
        assertThat(resultDs.value(0, wordGroupId.generateId("baz"))).isEqualTo(0)
    }

    @Test
    fun testDeserializationPreservesLabels() {
        val srcId = ColumnId.create<List<String>>("words")
        val wordGroupId = ColumnIdGroup.create<Int>("wordCounts")
        val wcLabel = "wc"
        val addCLabel = "addC"
        val dg = DataGraph.build {
            val src = setSource {
                columnIds += srcId
            }

            // This is the transform that uses a ColumnIdGroup
            val wordCount = addTransform(src, WordCounts(srcId, wordGroupId))
            wordCount.label = wcLabel

            val addC = addTransform(wordCount, AddConstantTransform(1))
            addC.label = addCLabel

            result = addC
        }

        // Now serialize it.
        val serDeser = GraphSerDeser()
        val output = ByteArrayOutputStream()
        serDeser.serialize(dg, output)

        // And deserilize it
        val deserDg = serDeser.deserialize(ByteArrayInputStream(output.toByteArray()))

        // We now check the labels of the nodes. We ensure the right label is on the right node by checking the node
        // that provides the input data to the labeled node.
        val wcNodes = deserDg.allNodes.filter { it!!.label == wcLabel }
        assertThat(wcNodes).hasSize(1)
        val wcNode = wcNodes[0]!!
        assertThat(wcNode.sources).hasSize(1)
        assertThat(wcNode.sources[0].source).isSameAs(deserDg.source)

        val addCNodes = deserDg.allNodes.filter { it!!.label == addCLabel }
        assertThat(addCNodes).hasSize(1)
        val addCNode = addCNodes[0]!!
        assertThat(addCNode.sources).hasSize(1)
        assertThat(addCNode.sources[0].source).isSameAs(wcNode)
    }

    @Test
    fun testCanSerDeserSingleItemDataTransform() {
        val sourceCol = ColumnId.create<String>("foo")
        val source = SourceGraphNode.build(0) {
            columnIds += sourceCol
        }
        val toLower = LowerCaseTransform()

        val serDeser = GraphSerDeser()
        val output = ByteArrayOutputStream()
        serDeser.serializeTransform(toLower, output)
        log.debug("Serialized as: {}", output.toString())
        val input = ByteArrayInputStream(output.toByteArray())

        val deserialized = serDeser.deserializeTransform(
                null, StandardJsonFormat.MetaData(toLower), listOf(source), input)
        assertThat(deserialized).isInstanceOf(LowerCaseTransform::class.java)
    }

    // See comments in GraphExecutionTest.testComplexTrainOnlyGraphExecution
    @Test
    fun testComplexGraphDoesNotSerializeTrainOnly() {
        val g1 = Graph1()

        val trainMatrix = listOf(
                listOf("FOO", true),
                listOf("foo", false),
                listOf("bar", false),
                listOf("bip", true),
                listOf("baz", true),
                listOf("biZzLE", true),
                listOf("BIzZle", false)
        )

        val trainRes = g1.graph.trainTransform(
                g1.graph.createTrainingSource(trainMatrix), Executors.newFixedThreadPool(3)).get()
        assertThat(trainRes.numRows).isEqualTo(trainMatrix.size)
        assertThat(trainRes.numColumns).isEqualTo(1)
        assertThat(trainRes.column(g1.resultId)).containsExactly(true, true, false, false, false, true, true)

        val serDeser = GraphSerDeser()

        // Serialize it
        val output = ByteArrayOutputStream()
        serDeser.serialize(g1.graph, output)

        // And deserilize it
        val deserDg = serDeser.deserialize(ByteArrayInputStream(output.toByteArray()))

        // Make sure only the non-training nodes were serialized
        assertThat(deserDg.allNodes.count { it != null }).isEqualTo(6)

        // And make sure it works.
        val testMatrix = listOf(
                listOf("FoO"),
                listOf("bar"),
                listOf("baZ"),
                listOf("bizzle"))
        val predictRes = g1.graph.transform(g1.graph.createSource(testMatrix), Executors.newFixedThreadPool(2)).get()
        assertThat(predictRes.numRows).isEqualTo(testMatrix.size)
        assertThat(predictRes.numColumns).isEqualTo(1)
        assertThat(predictRes.column(g1.resultId)).containsExactly(true, false, false, true)
    }

    @Test
    fun testSimpleInjection() {
        val sourceColId = ColumnId.create<Int>("src")
        val amountToAdd1 = 1232
        val amountToAdd2 = 4
        val injectionLabel = "INJECT"
        val dg = DataGraph.build {
            val source = setSource {
                columnIds += sourceColId
            }

            val trans1 = addTransform(
                    source, AddConstantTransform(amountToAdd1))
            trans1.label = injectionLabel

            val trans2 = addTransform(
                    trans1, AddConstantTransform(amountToAdd2))
            result = trans2
        }

        val serDeser = GraphSerDeser()
        val outStream = ByteArrayOutputStream(0)
        serDeser.serialize(dg, outStream)

        val amountToAddAfterInjection = 92

        // Will create another AddConstantTransform but with a different constant
        val factory = object : TransformFactory {
            override fun deserialize(
                    metaData: FormatMetaData,
                    sources: List<GraphNode>, serDeser: GraphSerDeser, input: InputStream): DataTransform {
                if (metaData is StandardJsonFormat.MetaData) {
                    assertThat(metaData.transformClass).isEqualTo(AddConstantTransform::class.java)
                    val theData = JsonMapper.mapper.readTree(input)
                    assertThat(theData.isObject).isTrue()
                    assertThat(theData.findValue("toAdd").intValue()).isEqualTo(amountToAdd1)
                    assertThat(sources).hasSize(1)

                    return AddConstantTransform(amountToAddAfterInjection)
                } else {
                    throw IllegalStateException("Unexpected metadata class: ${metaData.javaClass}")
                }
            }
        }

        serDeser.registerFactoryForLabel(injectionLabel, factory)

        // Now deserialize the thing....
        val inStream = ByteArrayInputStream(outStream.toByteArray())
        val deserGraph = serDeser.deserialize(inStream)

        val sourceValue = 18
        val sourceObs = deserGraph.createSource(sourceValue)
        val result = deserGraph.transform(sourceObs, Executors.newSingleThreadExecutor()).get()

        assertThat(result.value(0, sourceColId)).isEqualTo(sourceValue + amountToAddAfterInjection + amountToAdd2)
    }
}
