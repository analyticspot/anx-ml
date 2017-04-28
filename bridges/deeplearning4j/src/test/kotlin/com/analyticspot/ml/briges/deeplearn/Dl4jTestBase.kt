package com.analyticspot.ml.briges.deeplearn

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.testng.annotations.BeforeClass

/**
 * DeepLearning4j requires global configuration so we do that here and inherit it in all our DL4j tests.
 */
open class Dl4jTestBase {
    @BeforeClass
    fun globalSetup() {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    }
}
