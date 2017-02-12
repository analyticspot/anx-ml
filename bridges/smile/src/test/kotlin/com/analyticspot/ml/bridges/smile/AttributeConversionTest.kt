package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test
import smile.data.Attribute
import smile.data.NominalAttribute

class AttributeConversionTest {
    @Test
    fun testCategoricalConversion() {
        val catId = ColumnId.create<String>("foo")
        val catMeta = CategoricalFeatureMetaData(true, setOf("a", "b", "c"))

        val smileNominal = AttributeConversion.toAttribute(catId, catMeta)

        assertThat(smileNominal.isOpen).isFalse()
        assertThat(smileNominal.size()).isEqualTo(catMeta.possibleValues.size)
        assertThat(smileNominal.values()).containsExactlyElementsOf(catMeta.possibleValues)
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

        assertThat(smileNominal.toString(valA)).isEqualTo("a")
        assertThat(smileNominal.toString(valB)).isEqualTo("b")
        assertThat(smileNominal.toString(valC)).isEqualTo("c")
    }

    @Test
    fun testBooleanConversion() {
        val boolId = ColumnId.create<Boolean>("bool")

        val smileBoolean = AttributeConversion.toAttribute(boolId)

        assertThat(smileBoolean.isOpen).isFalse()
        assertThat(smileBoolean.size()).isEqualTo(2)
        assertThat(smileBoolean.values()).containsAll(listOf("true", "false"))
        assertThat(smileBoolean.values()).containsOnly("true", "false")
        assertThat(smileBoolean.name).isEqualTo(boolId.name)

        // We want true to be encoded as 1 in the smile data and false as 0 for consistency with what most people
        // expect.
        assertThat(smileBoolean.valueOf("true")).isEqualTo(1.0)
        assertThat(smileBoolean.valueOf("false")).isEqualTo(0.0)

        assertThat(smileBoolean.toString(1.0)).isEqualTo("true")
        assertThat(smileBoolean.toString(0.0)).isEqualTo("false")
    }

    @Test
    fun testDataSetConversion() {
        val catId = ColumnId.create<String>("foo")
        val catMeta = CategoricalFeatureMetaData(true, setOf("a", "b", "c"))
        val numId1 = ColumnId.create<Int>("bar")
        val numId2 = ColumnId.create<Double>("baz")
        val boolId = ColumnId.create<Boolean>("zap")

        val ds = DataSet.build {
            addColumn(catId, listOf("a", "a"), catMeta)
            addColumn(numId1, listOf(1, 2))
            addColumn(numId2, listOf(3.0, 4.0))
            addColumn(boolId, listOf(true, false))
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
        assertThat(nominalAttr.values()).containsExactlyElementsOf(catMeta.possibleValues)

        assertThat(attrs[3].name).isEqualTo("zap")
        assertThat(attrs[3].type).isEqualTo(Attribute.Type.NOMINAL)
        val boolAttr = attrs[3] as NominalAttribute
        assertThat(boolAttr.values()).contains("true", "false")
        assertThat(boolAttr.values()).containsOnly("true", "false")
    }
}
