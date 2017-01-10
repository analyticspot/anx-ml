package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.AddConstantTransform
import com.analyticspot.ml.framework.datagraph.DataGraph
import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datagraph.SourceGraphNode
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.dataset.SingleObservationDataSet
import com.analyticspot.ml.framework.datatransform.DataTransform
import com.analyticspot.ml.framework.datatransform.MergeTransform
import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueIdGroup
import com.analyticspot.ml.framework.observation.SingleValueObservation
import com.analyticspot.ml.framework.testutils.Graph1
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
        val valIdToTransform = ValueId.create<Int>("val2")
        val source = SourceGraphNode.build(0) {
            valueIds += listOf(ValueId.create<Int>("val1"), valIdToTransform)
        }

        val srcToken = source.token(valIdToTransform)

        // Construct the transform
        val amountToAdd = 11
        val resultId = ValueId.create<Int>("result")
        val trans = AddConstantTransform(amountToAdd, srcToken, resultId)

        val serDeser = GraphSerDeser()

        // Serialize it to the output stream
        val output = ByteArrayOutputStream()
        serDeser.serializeTransform(trans, output)
        log.debug("Transform serialized as: {}", output.toString())

        val input = ByteArrayInputStream(output.toByteArray())

        // Construct a new source with only 1 token. Note that the index of this token will be different than the one
        // from the original source.
        val newSource = SourceGraphNode.build(0) {
            valueIds += valIdToTransform
        }
        // Now deserialize relative to this new source
        val deserialized = serDeser.deserializeTransform(
                null, StandardJsonFormat.MetaData(trans), listOf(newSource), input)

        assertThat(deserialized).isInstanceOf(AddConstantTransform::class.java)
        val deserializedAddConstant = deserialized as AddConstantTransform
        assertThat(deserializedAddConstant.toAdd).isEqualTo(amountToAdd)
        assertThat(deserializedAddConstant.srcToken).isInstanceOf(IndexValueToken::class.java)
        assertThat(deserializedAddConstant.srcToken.name).isEqualTo(valIdToTransform.name)
        assertThat(deserializedAddConstant.srcToken.clazz).isEqualTo(valIdToTransform.clazz)
        assertThat((deserializedAddConstant.srcToken as IndexValueToken<Int>).index).isEqualTo(0)
    }

    @Test
    fun testCanSerializeAndDeserializeMerge() {
        val v1Id = ValueId.create<String>("v1")
        // Note: Not using sequential numbers here to ensure things deserialize using the correct data sets.
        val s1 = SourceGraphNode.build(12) {
            valueIds += v1Id
        }

        val v2Id = ValueId.create<String>("v2")
        val s2 = SourceGraphNode.build(8) {
            valueIds += v2Id
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
        val deserMerge = deserialized as MergeTransform
        assertThat(deserMerge.description.tokens.map { it.name }).isEqualTo(listOf("v1", "v2"))
    }

    @Test
    fun testCanSerializeAndDeserializeSimpleGraph() {
        val sourceValId = ValueId.create<Int>("src")
        val transformValId = ValueId.create<Int>("resultVal")
        val amountToAdd = 1232
        val dg = DataGraph.build {
            val source = setSource {
                valueIds += sourceValId
            }

            val trans = addTransform(
                    source, AddConstantTransform(amountToAdd, source.token(sourceValId), transformValId))
            result = trans
        }

        val serDeser = GraphSerDeser()
        val outStream = ByteArrayOutputStream(0)
        serDeser.serialize(dg, outStream)

        // Now deserialize the thing....
        val inStream = ByteArrayInputStream(outStream.toByteArray())
        val deserGraph = serDeser.deserialize(inStream)

        assertThat(deserGraph.source.tokens).hasSize(1)
        assertThat(deserGraph.source.tokens[0].id).isEqualTo(sourceValId)

        val sourceValue = 18
        val sourceObs = deserGraph.buildSourceObservation(sourceValue)
        val result = deserGraph.transform(sourceObs, Executors.newSingleThreadExecutor()).get()

        val resultToken = deserGraph.result.token(transformValId)
        assertThat(result.value(resultToken)).isEqualTo(sourceValue + amountToAdd)
    }

    @Test
    fun testSourceGraphNodeSerializesTrainOnlyInformation() {
        val sourceValIds = listOf(
                ValueId.create<Int>("src1"),
                ValueId.create<String>("src2"))
        val trainOnlySourceValIds = listOf(
                ValueId.create<Boolean>("srcT1"),
                ValueId.create<Int>("srcT2"))
        val dg = DataGraph.build {
            val source = setSource {
                valueIds += sourceValIds
                trainOnlyValueIds += trainOnlySourceValIds
            }

            result = source
        }

        val serDeser = GraphSerDeser()
        val outStream = ByteArrayOutputStream(0)
        serDeser.serialize(dg, outStream)

        // Now deserialize the thing....
        val inStream = ByteArrayInputStream(outStream.toByteArray())
        val deserGraph = serDeser.deserialize(inStream)

        assertThat(deserGraph.source.tokens.map { it.id }).isEqualTo(sourceValIds.plus(trainOnlySourceValIds))
        assertThat(deserGraph.source.trainOnlyValueIds).isEqualTo(trainOnlySourceValIds)
    }

    @Test
    fun testTokenGroupsSerialize() {
        val srcId = ValueId.create<List<String>>("words")
        val wordGroupId = ValueIdGroup.create<Int>("wordCounts")
        val dg = DataGraph.build {
            val src = setSource {
                valueIds += srcId
            }

            // This is the transform that uses a ValueIdGroup/ValueTokenGroup.
            val wordCount = addTransform(src, WordCounts(src.token(srcId), wordGroupId))

            result = wordCount
        }

        // Now run the transform and see what comes out the other side.
        val sourceSet = IterableDataSet(listOf(
                SingleValueObservation.create(listOf("foo", "bar", "bar")),
                SingleValueObservation.create(listOf("bar", "baz", "bar"))
        ))

        // Train it.
        dg.trainTransform(sourceSet, Executors.newSingleThreadExecutor()).get()

        // Now serialize it.
        val serDeser = GraphSerDeser()
        val output = ByteArrayOutputStream()
        serDeser.serialize(dg, output)

        // And deserilize it
        val deserDg = serDeser.deserialize(ByteArrayInputStream(output.toByteArray()))

        val toTransform = SingleObservationDataSet(
                SingleValueObservation.create(listOf("foo", "bar", "foo", "foo"))
        )

        val resultDs = deserDg.transform(toTransform, Executors.newSingleThreadExecutor()).get().toList()

        assertThat(resultDs).hasSize(1)
        val firstRow = resultDs[0]
        // Note that in the following I'm relying on the fact that the words are assigned indices in the order that they
        // were encountered. Safe for the current implementation of the transform since that's just for testing.
        assertThat(firstRow.values(dg.result.tokenGroup(wordGroupId))).isEqualTo(listOf(3, 1, 0))
    }

    // See comments in GraphExecutionTest.testComplexTrainOnlyGraphExecution
    @Test
    fun testComplexGraphDoesNotSerializeTrainOnly() {
        val g1 = Graph1()

        val trainMatrix = listOf(
                g1.graph.buildSourceObservation("FOO", true),
                g1.graph.buildSourceObservation("foo", false),
                g1.graph.buildSourceObservation("bar", false),
                g1.graph.buildSourceObservation("bip", true),
                g1.graph.buildSourceObservation("baz", true),
                g1.graph.buildSourceObservation("biZzLE", true),
                g1.graph.buildSourceObservation("BIzZle", false)
        )

        val resultToken = g1.graph.result.token(g1.resultId)
        val trainRes = g1.graph.trainTransform(IterableDataSet(trainMatrix), Executors.newFixedThreadPool(3)).get()
        val trainRestList = trainRes.map { it.value(resultToken) }
        assertThat(trainRestList).isEqualTo(listOf(true, true, false, false, false, true, true))

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
                g1.graph.buildSourceObservation("FoO"),
                g1.graph.buildSourceObservation("bar"),
                g1.graph.buildSourceObservation("baZ"),
                g1.graph.buildSourceObservation("bizzle"))
        val predictRes = g1.graph.transform(IterableDataSet(testMatrix), Executors.newFixedThreadPool(2)).get()
        val predictResList = predictRes.map { it.value(resultToken) }
        assertThat(predictResList).isEqualTo(listOf(true, false, false, true))
    }

    @Test
    fun testSimpleInjection() {
        val sourceValId = ValueId.create<Int>("src")
        val transformValId = ValueId.create<Int>("resultVal")
        val amountToAdd1 = 1232
        val amountToAdd2 = 4
        val injectionLabel = "INJECT"
        val dg = DataGraph.build {
            val source = setSource {
                valueIds += sourceValId
            }

            val trans1 = addTransform(
                    source, AddConstantTransform(amountToAdd1, source.token(sourceValId), transformValId))
            trans1.label = injectionLabel

            val trans2 = addTransform(
                    trans1, AddConstantTransform(amountToAdd2, trans1.token(transformValId), transformValId))
            result = trans2
        }

        val serDeser = GraphSerDeser()
        val outStream = ByteArrayOutputStream(0)
        serDeser.serialize(dg, outStream)

        val amountToAddAfterInjection = 92

        // Will create another AddConstantTransform but with a different constant
        val factory = object : TransformFactory<StandardJsonFormat.MetaData> {
            override fun deserialize(
                    metaData: StandardJsonFormat.MetaData,
                    sources: List<GraphNode>, input: InputStream): DataTransform {
                assertThat(metaData.transformClass).isEqualTo(AddConstantTransform::class.java)
                val theData = JsonMapper.mapper.readTree(input)
                assertThat(theData.isObject).isTrue()
                assertThat(theData.findValue("toAdd").intValue()).isEqualTo(amountToAdd1)
                assertThat(sources).hasSize(1)

                return AddConstantTransform(amountToAddAfterInjection, sources[0].token(sourceValId), transformValId)
            }
        }

        serDeser.registerFactoryForLabel(injectionLabel, factory)

        // Now deserialize the thing....
        val inStream = ByteArrayInputStream(outStream.toByteArray())
        val deserGraph = serDeser.deserialize(inStream)

        val sourceValue = 18
        val sourceObs = deserGraph.buildSourceObservation(sourceValue)
        val result = deserGraph.transform(sourceObs, Executors.newSingleThreadExecutor()).get()

        val resultToken = deserGraph.result.token(transformValId)
        assertThat(result.value(resultToken)).isEqualTo(sourceValue + amountToAddAfterInjection + amountToAdd2)
    }
}
