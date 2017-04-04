package com.analyticspot.ml.framework.utils

import com.analyticspot.ml.framework.dataset.ListColumn
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class DataUtilsTest {
    @Test
    fun testEncodeAndDecodeCategorical() {
        val catCol = ListColumn<String>(listOf("a", "b", "a", "c", "a", "b", "c", "b"))

        val (encoded, intToStr) = DataUtils.encodeCategorical(catCol)

        // All values should have been mapped to [0, 2] since there's only 3 unique values in catCol
        assertThat(encoded.max()).isEqualTo(2)
        assertThat(encoded.min()).isEqualTo(0)

        // And we should be able to reverse the procedure
        val decoded = DataUtils.decodeCategorical(intToStr, encoded)
        assertThat(decoded).containsExactlyElementsOf(catCol)
    }
}
