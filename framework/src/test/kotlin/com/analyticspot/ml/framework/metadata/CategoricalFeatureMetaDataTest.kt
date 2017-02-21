package com.analyticspot.ml.framework.metadata

import com.analyticspot.ml.framework.dataset.ListColumn
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class CategoricalFeatureMetaDataTest {
    @Test
    fun testFromColumn() {
        val col1 = ListColumn<String>(listOf("foo", "bar", "baz", "bar", "foo", "bizzle", "foo"))
        val cat1 = CategoricalFeatureMetaData.fromStringColumn(col1)

        assertThat(cat1.possibleValues).containsExactly("foo", "bar", "baz", "bizzle")
        assertThat(cat1.maybeMissing).isFalse()

        val col2 = ListColumn<String?>(listOf("foo", "bar", "baz", "bar", "foo", "bizzle", "foo", null))
        val cat2 = CategoricalFeatureMetaData.fromStringColumn(col2)

        assertThat(cat2.possibleValues).containsExactly("foo", "bar", "baz", "bizzle")
        assertThat(cat2.maybeMissing).isTrue()
    }

    @Test
    fun testDoesColumnConform() {
        val col1 = ListColumn<String>(listOf("foo", "bar", "baz", "bar", "foo", "bizzle", "foo"))
        val metaData = CategoricalFeatureMetaData.fromStringColumn(col1)

        val col2 = ListColumn<String>(listOf("foo", "bar", "foo", "baz"))
        assertThat(metaData.doesColumnConformToMetaData(col2)).isTrue()

        val col3 = ListColumn<String>(listOf("foo", "bar", "foo", "baz", "not-in-col"))
        assertThat(metaData.doesColumnConformToMetaData(col3)).isFalse()
    }

    @Test
    fun testMakeColumnConformToMetaData() {
        val col1 = ListColumn<String>(listOf("foo", "bar", "baz", "bar", "foo", "bizzle", "foo"))
        val metaData = CategoricalFeatureMetaData.fromStringColumn(col1)

        val col2 = ListColumn<String>(listOf("foo", "bar", "foo", "baz"))
        assertThat(metaData.makeColumnConformToMetaData(col2)).containsExactlyElementsOf(col2)

        val col3 = ListColumn<String>(listOf("foo", "bar", "foo", "baz", "not-in-col"))
        assertThat(metaData.makeColumnConformToMetaData(col3)).containsExactly("foo", "bar", "foo", "baz", null)
    }
}
