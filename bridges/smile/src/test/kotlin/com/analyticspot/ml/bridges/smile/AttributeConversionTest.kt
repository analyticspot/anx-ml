package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.FeatureDataSet
import com.analyticspot.ml.framework.feature.CategoricalFeatureId
import com.analyticspot.ml.framework.feature.NumericalFeatureId
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test
import smile.data.Attribute
import smile.data.NominalAttribute

class AttributeConversionTest {
    @Test
    fun testCategoricalConversion() {
        val catId = CategoricalFeatureId("foo", true, setOf("a", "b", "c"))

        val smileNominal = AttributeConversion.toAttribute(catId)

        assertThat(smileNominal.isOpen).isFalse()
        assertThat(smileNominal.size()).isEqualTo(catId.possibleValues.size)
        assertThat(smileNominal.values()).containsExactlyElementsOf(catId.possibleValues)
        assertThat(smileNominal.name).isEqualTo(catId.name)

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

    @Test
    fun testDataSetConversion() {
        val catId = CategoricalFeatureId("foo", true, setOf("a", "b", "c"))
        val numId1 = NumericalFeatureId("bar", true)
        val numId2 = NumericalFeatureId("baz", false)

        val ds = FeatureDataSet.build {
            addColumn(catId, listOf("a", "a"))
            addColumn(numId1, listOf(1.0, 2.0))
            addColumn(numId2, listOf(3.0, 4.0))
        }

        val attrs = AttributeConversion.toSmileAttributes(ds)

        assertThat(attrs).hasSize(ds.numColumns)

        // Make sure the attributes are correctly ordered
        assertThat(attrs.map { it.name }).isSorted()

        // The first attribute should be "bar" since they're sorted
        assertThat(attrs[0].name).isEqualTo("bar")
        assertThat(attrs[0].type).isEqualTo(Attribute.Type.NUMERIC)

        assertThat(attrs[1].name).isEqualTo("baz")
        assertThat(attrs[1].type).isEqualTo(Attribute.Type.NUMERIC)

        assertThat(attrs[2].name).isEqualTo("foo")
        assertThat(attrs[2].type).isEqualTo(Attribute.Type.NOMINAL)
        val nominalAttr = attrs[2] as NominalAttribute
        assertThat(nominalAttr.values()).containsExactlyElementsOf(catId.possibleValues)
    }
}
