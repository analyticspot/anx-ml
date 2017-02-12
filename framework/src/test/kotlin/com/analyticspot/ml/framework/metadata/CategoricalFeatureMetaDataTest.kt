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
}
