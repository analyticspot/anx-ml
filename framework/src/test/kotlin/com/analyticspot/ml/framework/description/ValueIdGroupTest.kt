package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.serialization.JsonMapper
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class ValueIdGroupTest {
    @Test
    fun testCanSerializeAndDeserialize() {
        val v = ValueIdGroup.create<Int>("foo")

        val serialized = JsonMapper.mapper.writeValueAsString(v)

        val deser = JsonMapper.mapper.readValue(serialized, ValueIdGroup::class.java)

        assertThat(deser).isEqualTo(v)
    }
}
