package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.feature.CategoricalFeatureId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class AttributeConversionTest {
    @Test
    fun testCategoricalConversion() {
        val catId = CategoricalFeatureId("foo", true, setOf("a", "b", "c"))

        val smileNominal = AttributeConversion.toAttribute(catId)

        assertThat(smileNominal.isOpen).isFalse()
        assertThat(smileNominal.size()).isEqualTo(catId.possibleValues.size)
        assertThat(smileNominal.values()).containsExactlyElementsOf(catId.possibleValues)

        // Converting categorical values to double (which is how they'll be encoded in the double arrays passed to
        // smile algorithms) must work. So passing in the same value should yeild the same result and passing in
        // different values should return different results.
        val valA = smileNominal.valueOf("a")
        val valB = smileNominal.valueOf("b")
        val valC = smileNominal.valueOf("c")

        assertThat(valA).isNotEqualTo(valB)
        assertThat(valA).isNotEqualTo(valC)
        assertThat(valB).isNotEqualTo(valC)
        assertThat(valA).isEqualTo(smileNominal.valueOf("a"))
        assertThat(valB).isEqualTo(smileNominal.valueOf("b"))
        assertThat(valC).isEqualTo(smileNominal.valueOf("c"))
    }
}
