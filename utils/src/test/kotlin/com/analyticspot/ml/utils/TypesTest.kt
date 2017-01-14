package com.analyticspot.ml.utils

import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

@Suppress("PLATFORM_CLASS_MAPPED_TO_KOTLIN")
class TypesTest {
    @Test
    fun testKClassIsAssignableFrom() {
        assertThat(Integer::class isAssignableFrom Integer::class).isTrue()
        assertThat(Integer::class isAssignableFrom Int::class).isTrue()
        assertThat(Int::class isAssignableFrom Integer::class).isTrue()
        assertThat(Int::class isAssignableFrom Int::class).isTrue()

        assertThat(Integer::class isAssignableFrom Double::class).isFalse()
        assertThat(Integer::class isAssignableFrom java.lang.Double::class).isFalse()
        assertThat(Int::class isAssignableFrom Double::class).isFalse()
        assertThat(Int::class isAssignableFrom java.lang.Double::class).isFalse()
        assertThat(Int::class isAssignableFrom String::class).isFalse()
        assertThat(Integer::class isAssignableFrom String::class).isFalse()

        assertThat(String::class isAssignableFrom String::class).isTrue()
        assertThat(String::class isAssignableFrom Object::class).isFalse()
        assertThat(Object::class isAssignableFrom String::class).isTrue()
    }

    @Test
    fun testJavaClassIsAssignableFrom() {
        assertThat(Integer::class.java isAssignableFrom Integer::class.java).isTrue()
        assertThat(Integer::class.java isAssignableFrom Int::class.java).isTrue()
        assertThat(Int::class.java isAssignableFrom Integer::class.java).isTrue()
        assertThat(Int::class.java isAssignableFrom Int::class.java).isTrue()

        assertThat(Integer::class.java isAssignableFrom Double::class.java).isFalse()
        assertThat(Integer::class.java isAssignableFrom java.lang.Double::class.java).isFalse()
        assertThat(Int::class.java isAssignableFrom Double::class.java).isFalse()
        assertThat(Int::class.java isAssignableFrom java.lang.Double::class.java).isFalse()
        assertThat(Int::class.java isAssignableFrom String::class.java).isFalse()
        assertThat(Integer::class.java isAssignableFrom String::class.java).isFalse()

        assertThat(String::class.java isAssignableFrom String::class.java).isTrue()
        assertThat(String::class.java isAssignableFrom Object::class.java).isFalse()
        assertThat(Object::class.java isAssignableFrom String::class.java).isTrue()
    }
}
