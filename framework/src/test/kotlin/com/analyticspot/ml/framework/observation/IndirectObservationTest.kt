package com.analyticspot.ml.framework.observation

import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.IndirectValueToken
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueToken
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class IndirectObservationTest {
    @Test
    fun testObservationIndexingWorks() {
        val o1 = SingleValueObservation.create("Hello")
        val o2 = SingleValueObservation.create(11)
        val o3 = ArrayObservation.create(22.22, "Goodbye")

        val merged = IndirectObservation(listOf(o1, o2, o3))

        val t1 = IndirectValueToken(0, ValueToken(ValueId.create<String>("v1")))
        assertThat(merged.value(t1)).isEqualTo("Hello")

        val t2 = IndirectValueToken(1, ValueToken(ValueId.create<Int>("v2")))
        assertThat(merged.value(t2)).isEqualTo(11)

        val o3Idx1 = IndexValueToken.create(0, ValueId.create<Double>("o3.1"))
        val t3 = IndirectValueToken(2, o3Idx1)
        assertThat(merged.value(t3)).isEqualTo(22.22)

        val o3Idx2 = IndexValueToken.create(1, ValueId.create<String>("o3.2"))
        val t4 = IndirectValueToken(2, o3Idx2)
        assertThat(merged.value(t4)).isEqualTo("Goodbye")
    }
}
