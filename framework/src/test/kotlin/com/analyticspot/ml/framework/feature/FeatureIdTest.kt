package com.analyticspot.ml.framework.feature

import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.serialization.JsonMapper
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test

class FeatureIdTest {
    companion object {
        private val log = LoggerFactory.getLogger(FeatureIdTest::class.java)
    }

    // Test that we can serialize and deserialize ColumnId, FeatureId, and the subclasses of FeatureId properly.
    @Test
    fun testCanSerDeserAllSuperAndSubclasses() {
        val colId = ColumnId.create<String>("foo")
        val colIdSer = JsonMapper.mapper.writeValueAsString(colId)
        log.debug("Serialized as: {}", colIdSer)
        val colIdD = JsonMapper.mapper.readValue(colIdSer, ColumnId::class.java)
        assertThat(colIdD.name).isEqualTo(colId.name)
        assertThat(colIdD.clazz).isEqualTo(colId.clazz)

        val boolId = BooleanFeatureId("boo", true)
        val boolIdSer = JsonMapper.mapper.writeValueAsString(boolId)
        log.debug("Serialized as: {}", boolIdSer)
        val boolIdD = JsonMapper.mapper.readValue(boolIdSer, ColumnId::class.java)
        assertThat(boolIdD.name).isEqualTo(boolId.name)
        assertThat(boolIdD.clazz).isEqualTo(boolId.clazz)
        assertThat(boolIdD).isInstanceOf(BooleanFeatureId::class.java)
        val boolIdCast = boolIdD as BooleanFeatureId
        assertThat(boolIdCast.maybeMissing).isEqualTo(true)

        val catId = CategoricalFeatureId("boo", true, setOf("foo", "bar", "baz"))
        val catIdSer = JsonMapper.mapper.writeValueAsString(catId)
        log.debug("Serialized as: {}", catIdSer)
        val catIdD = JsonMapper.mapper.readValue(catIdSer, ColumnId::class.java)
        assertThat(catIdD.name).isEqualTo(catId.name)
        assertThat(catIdD.clazz).isEqualTo(catId.clazz)
        assertThat(catIdD).isInstanceOf(CategoricalFeatureId::class.java)
        val catIdCast = catIdD as CategoricalFeatureId
        assertThat(catIdCast.maybeMissing).isEqualTo(true)
        assertThat(catIdCast.possibleValues).isEqualTo(setOf("foo", "bar", "baz"))
    }

}
