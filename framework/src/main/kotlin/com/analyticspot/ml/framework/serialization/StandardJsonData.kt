package com.analyticspot.ml.framework.serialization

import com.analyticspot.ml.framework.datatransform.DataTransform
import com.fasterxml.jackson.annotation.JsonProperty

/**
 *
 */
class StandardJsonData(@JsonProperty("transformClass") val transformClass: Class<out DataTransform>) : FormatData() {
}
