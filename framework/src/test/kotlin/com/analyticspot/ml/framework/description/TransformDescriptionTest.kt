package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.serialization.JsonMapper
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test

class TransformDescriptionTest {
    companion object {
        private val log = LoggerFactory.getLogger(TransformDescriptionTest::class.java)
    }

    @Test
    fun testCanSerializeAndDeserialize() {
        val td = TransformDescription(
                listOf(ColumnId.create<String>("c1"), ColumnId.create<Int>("c2")),
                listOf(ColumnIdGroup.create<String>("g1"), ColumnIdGroup.create<Double>("c2"))
        )

        val serialized = JsonMapper.mapper.writeValueAsString(td)
        log.debug("Serialized as: {}", serialized)
        val deserialized = JsonMapper.mapper.readValue(serialized, TransformDescription::class.java)

        assertThat(deserialized).isEqualTo(td)
    }
}
