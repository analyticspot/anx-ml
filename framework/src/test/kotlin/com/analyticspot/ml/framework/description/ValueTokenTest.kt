package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.datagraph.GraphNode
import com.analyticspot.ml.framework.datagraph.SourceGraphNode
import com.analyticspot.ml.framework.serialization.JsonMapper
import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.InjectableValues
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test

class ValueTokenTest {
    companion object {
        private val log = LoggerFactory.getLogger(ValueTokenTest::class.java)
    }

    @Test
    fun testSerializeAndDeserialize() {
        val valId = ValueId.create<String>("foo")
        val source = SourceGraphNode.build(0) {
            valueIds += valId
        }
        val token = source.token(valId)
        val serialized = JsonMapper.mapper.writeValueAsString(token)
        log.debug("Serialized token is: {}", serialized)

        // Create a new source with different indices for the tokens to make sure we're deserializing relative to that
        // source.
        val v0 = ValueId.create<String>("bar")
        val v1 = ValueId.create<String>("baz")
        val v3 = ValueId.create<String>("bap")
        val newSource = SourceGraphNode.build(1) {
            valueIds += listOf(v0, v1, valId, v3)
        }

        val injectables = InjectableValues.Std().addValue(GraphNode::class.java, newSource)

        val reader = JsonMapper.mapper.readerFor(object : TypeReference<ValueToken<String>>() {}).with(injectables)

        val deserializedToken = reader.readValue<ValueToken<String>>(serialized)
        assertThat(deserializedToken).isInstanceOf(IndexValueToken::class.java)
        val tokAsIndex = deserializedToken as IndexValueToken<String>
        assertThat(tokAsIndex.index).isEqualTo(2)
    }
}
