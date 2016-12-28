package com.analyticspot.ml.framework.observation

import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class ObservationUtilsTest {
    @Test
    fun testEqualValuesWithDifferentObservationTypes() {
        val a = ArrayObservation.create("hello", 11)
        val b = SingleValueObservation.create(11)
        val c = SingleValueObservation.create(12)
        val d = SingleValueObservation.create("hello")
        val e = SingleValueObservation.create("goodbye")

        // Note: This same token will work for both types of observations
        val numberTok = IndexValueToken.create(1, ValueId.create<Int>("number"))
        val stringTok = IndexValueToken.create(0, ValueId.create<String>("string"))

        assertThat(equalValues(listOf(numberTok), a, b)).isTrue()
        assertThat(equalValues(listOf(numberTok), a, c)).isFalse()
        assertThat(equalValues(listOf(numberTok), b, c)).isFalse()

        assertThat(equalValues(listOf(stringTok), a, d)).isTrue()
        assertThat(equalValues(listOf(stringTok), a, e)).isFalse()
        assertThat(equalValues(listOf(stringTok), d, e)).isFalse()
    }

    @Test
    fun testEqualValuesWithMutliValuedObservations() {
        val a = ArrayObservation.create("hello", 11, 12.2)
        val b = ArrayObservation.create("hello", 11, 12.2)
        val c = ArrayObservation.create("hello", 11, 12.3)

        val tokens = listOf(
                IndexValueToken.create(0, ValueId.create<String>("v1")),
                IndexValueToken.create(1, ValueId.create<Int>("v2")),
                IndexValueToken.create(2, ValueId.create<Double>("v3")))
        assertThat(equalValues(tokens, a, b)).isTrue()
        assertThat(equalValues(tokens, a, c)).isFalse()
    }
}
