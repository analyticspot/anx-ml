package com.analyticspot.ml.framework.dataset

import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.observation.ArrayObservation
import org.assertj.core.api.Assertions.assertThat
import org.testng.annotations.Test

class SingleObservationDataSetTest {
    @Test
    fun testSingleObservationDataSet() {
        val arrayObs = ArrayObservation.create("Hello", 18)
        val singleObsDs = SingleObservationDataSet(arrayObs)

        val allObs = singleObsDs.map { it }

        assertThat(allObs).hasSize(1)
        val theObs = allObs[0]
        assertThat(theObs.value(IndexValueToken.create(0, ValueId.create<String>("foo")))).isEqualTo("Hello")
        assertThat(theObs.value(IndexValueToken.create(1, ValueId.create<Int>("bar")))).isEqualTo(18)
    }
}
