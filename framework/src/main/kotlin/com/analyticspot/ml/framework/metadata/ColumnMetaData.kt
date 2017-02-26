package com.analyticspot.ml.framework.metadata

import com.fasterxml.jackson.annotation.JsonTypeInfo
import com.fasterxml.jackson.annotation.JsonTypeInfo.As
import com.fasterxml.jackson.annotation.JsonTypeInfo.Id

/**
 * Interface for meta-data about a column. Metadata can be anything but typically includes things like the
 * range of legal values for a column, etc. We often include metadata for categorical features for example.
 */
@JsonTypeInfo(use= Id.CLASS, include= As.PROPERTY, property="class")
interface ColumnMetaData

/**
 * Metadata that includes a boolean indicating if the column can have missing values or not.
 */
open class MaybeMissingMetaData(val maybeMissing: Boolean) : ColumnMetaData {
}
