package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.AddConstantTransform
import com.analyticspot.ml.framework.datagraph.SourceGraphNode
import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream

class StandardJsonFormatModuleTest {

    companion object {
        private val log = LoggerFactory.getLogger(StandardJsonFormatModuleTest::class.java)
    }

    // Serialize a simple AddConstantTransform and then deserialize it with the factory. Changes the source Graphnode so
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

        val module = StandardJsonFormatModule()

        // Serialize it to the output stream
        val output = ByteArrayOutputStream()
        module.serialize(trans, output)
        log.debug("Transform serialized as: {}", output.toString())

        val input = ByteArrayInputStream(output.toByteArray())
        val factory = module.getFactory(null)

        // Construct a new source with only 1 token. Note that the index of this token will be different than the one
        // from the original source.
        val newSource = SourceGraphNode.build(0) {
            valueIds += valIdToTransform
        }
        // Now deserialize relative to this new source
        val deserialized = factory.deserialize(StandardJsonData(trans.javaClass), listOf(newSource), input)

        assertThat(deserialized).isInstanceOf(AddConstantTransform::class.java)
        val deserializedAddConstant = deserialized as AddConstantTransform
        assertThat(deserializedAddConstant.toAdd).isEqualTo(amountToAdd)
        assertThat(deserializedAddConstant.srcToken).isInstanceOf(IndexValueToken::class.java)
        assertThat(deserializedAddConstant.srcToken.name).isEqualTo(valIdToTransform.name)
        assertThat(deserializedAddConstant.srcToken.clazz).isEqualTo(valIdToTransform.clazz)
        assertThat((deserializedAddConstant.srcToken as IndexValueToken<Int>).index).isEqualTo(0)
    }
}

