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

    @Test
    fun testCanSerializeAndDeserializeSimpleTransform() {
        val valIdToTransform = ValueId.create<Int>("val2")
        val source = SourceGraphNode.build(0) {
            valueIds += listOf(ValueId.create<Int>("val1"), valIdToTransform)
        }

        val srcToken = source.token(valIdToTransform)

        val amountToAdd = 11
        val resultId = ValueId.create<Int>("result")
        val trans = AddConstantTransform(amountToAdd, srcToken, resultId)

        val module = StandardJsonFormatModule()

        val output = ByteArrayOutputStream()
        module.serialize(trans, output)
        log.debug("Transform serialized as: {}", output.toString())

        val input = ByteArrayInputStream(output.toByteArray())
        val factory = module.getFactory(null)

        val newSource = SourceGraphNode.build(0) {
            valueIds += valIdToTransform
        }
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

