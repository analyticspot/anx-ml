package com.analyticspot.ml.framework.metadata

/**
 * Interface for meta-data about a column. Metadata can be anything but typically includes things like the
 * range of legal values for a column, etc. We often include metadata for categorical features for example.
 */
interface ColumnMetaData

/**
 * Metadata that includes a boolean indicating if the column can have missing values or not.
 */
open class MaybeMissingMetaData(val maybeMissing: Boolean) : ColumnMetaData {
}
