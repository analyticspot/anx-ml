package com.analyticspot.ml.framework.serialization

import com.fasterxml.jackson.annotation.JsonProperty

/**
 *
 */
class StandardJsonData(@JsonProperty("transformClass") private val transformClass: String) : FormatModuleData() {
}
