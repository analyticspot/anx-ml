package com.analyticspot.ml.bridges.smile

import com.analyticspot.ml.framework.feature.CategoricalFeatureId
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import smile.classification.ClassifierTrainer
import smile.classification.DecisionTree

class SmileClassifierTest {
    companion object {
        private val log = LoggerFactory.getLogger(SmileClassifierTest::class.java)
    }

    @Test
    fun testSimpleClassifier() {
        val dtTrainer = DecisionTree.Trainer()
        log.info("Type of dtTrainer is: {}", DecisionTree.Trainer::class)
        val targetId = CategoricalFeatureId("foo", false, setOf("a", "b", "c"))

        val trans = SmileClassifier(targetId, dtTrainer)
    }
}
