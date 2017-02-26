package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.serialization.JsonMapper
import com.fasterxml.jackson.core.type.TypeReference
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.io.ByteArrayOutputStream

class ColumnTest {
    companion object {
        private val log = LoggerFactory.getLogger(ColumnTest::class.java)
    }

    // This assumes that the data stored by the column is a serializable type.
    @Test
    fun testCanSerializeAndDeserialize() {
        val c = ListColumn(listOf("foo", "bar", "baz"))

        val serialized = JsonMapper.mapper.writeValueAsString(c)
        log.debug("Column serialized as: {}", serialized)

        val deser: Column<String> = JsonMapper.mapper.readValue(serialized,
                object : TypeReference<Column<String>>() {})

        assertThat(deser.size).isEqualTo(c.size)
        assertThat(deser.toList()).isEqualTo(c.toList())
    }
}
