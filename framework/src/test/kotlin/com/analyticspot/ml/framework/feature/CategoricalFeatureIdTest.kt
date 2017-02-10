package com.analyticspot.ml.framework.feature

import com.analyticspot.ml.framework.dataset.ListColumn
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class CategoricalFeatureIdTest {
    @Test
    fun testFromColumn() {
        val col1 = ListColumn<String>(listOf("foo", "bar", "baz", "bar", "foo", "bizzle", "foo"))
        val cat1 = CategoricalFeatureId.fromStringColumn(col1, "thename")

        assertThat(cat1.name).isEqualTo("thename")
        assertThat(cat1.possibleValues).containsExactly("foo", "bar", "baz", "bizzle")
        assertThat(cat1.maybeMissing).isFalse()

        val col2 = ListColumn<String?>(listOf("foo", "bar", "baz", "bar", "foo", "bizzle", "foo", null))
        val cat2 = CategoricalFeatureId.fromStringColumn(col2, "thename")

        assertThat(cat2.name).isEqualTo("thename")
        assertThat(cat2.possibleValues).containsExactly("foo", "bar", "baz", "bizzle")
        assertThat(cat2.maybeMissing).isTrue()
    }
}
