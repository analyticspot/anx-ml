package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datagraph.AddConstantTransform
import com.analyticspot.ml.framework.datagraph.SourceGraphNode
import com.analyticspot.ml.framework.description.ValueId
import org.slf4j.LoggerFactory
import org.testng.annotations.Test

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

        val serialized = JsonMapper.mapper.writeValueAsString(trans)
        log.info("Transform serialized as: {}", serialized)
    }

}

