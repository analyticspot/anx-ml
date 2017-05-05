package com.analyticspot.ml.framework.description

import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class ColumnIdGroupTest {
    @Test
    fun testIsInGroup() {
        val g1 = ColumnIdGroup.create<String>("g1")
        val g2 = ColumnIdGroup.create<String>("g2")

        val c1g1 = g1.generateId("foo")
        assertThat(g1.isInGroup(c1g1)).isTrue()
        assertThat(g2.isInGroup(c1g1)).isFalse()

        val c2g1 = g1.generateId("bazzle")
        assertThat(g1.isInGroup(c2g1)).isTrue()
        assertThat(g2.isInGroup(c2g1)).isFalse()

        val c1g2 = g2.generateId("foo")
        assertThat(g2.isInGroup(c1g2)).isTrue()
        assertThat(g1.isInGroup(c1g2)).isFalse()
    }
}
