package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.AddConstantTransform
import com.analyticspot.ml.framework.datagraph.SourceGraphNode
import com.analyticspot.ml.framework.datatransform.MergeTransform
import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream

/**

 */
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
}
