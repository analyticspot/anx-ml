package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import org.slf4j.LoggerFactory
import org.testng.annotations.Test

class ComputationGraphTransformTest {
    companion object {
        private val log = LoggerFactory.getLogger(ComputationGraphTransformTest::class.java)
    }

    @Test
    fun testCanTrainSimpleMlp() {
        val ds = DataSet.fromSaved(javaClass.getResourceAsStream("/iris.data.json"))
        log.info("DataSet: {}", ds.saveToString())

        val (trainDs, validDs) = ds.randomSubsets(0.75f)

        val trainFeatures = trainDs.allColumnsExcept("target")
    }
}
