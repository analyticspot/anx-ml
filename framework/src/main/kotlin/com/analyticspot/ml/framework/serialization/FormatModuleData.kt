package com.analyticspot.ml.framework.serialization

import com.fasterxml.jackson.annotation.JsonTypeInfo

/**
 * Base class for all [FormatModuleData] subclasses. This does little besides serialize the subclass so we know which
 * subclass to deserialize.
 */
@JsonTypeInfo(use= JsonTypeInfo.Id.CLASS, include= JsonTypeInfo.As.PROPERTY, property="type")
open class FormatModuleData {

}
