package com.analyticspot.ml.framework.serialization

import com.fasterxml.jackson.annotation.JsonTypeInfo
import com.fasterxml.jackson.annotation.JsonTypeInfo.As
import com.fasterxml.jackson.annotation.JsonTypeInfo.Id

/**
 *
 */
@JsonTypeInfo(use= Id.CLASS, include= As.PROPERTY, property="class")
interface FormatMetaData {
    /**
     * The format associated with this metadata.
     */
    val formatClass: Class<out Format<*>>
}
