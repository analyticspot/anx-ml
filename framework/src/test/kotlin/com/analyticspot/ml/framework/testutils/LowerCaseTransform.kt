package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.datatransform.SingleItemDataTransform
import com.analyticspot.ml.framework.description.TransformDescription

/**
 * Converts all the `String` type columns in a [DataSet] to lowercase.
 */
class LowerCaseTransform(srcDescription: TransformDescription)
    : SingleItemDataTransform<String, String>(srcDescription, String::class, String::class) {

    override fun transformItem(input: String): String {
        return input.toLowerCase()
    }
}
