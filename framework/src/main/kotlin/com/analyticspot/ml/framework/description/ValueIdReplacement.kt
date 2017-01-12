package com.analyticspot.ml.framework.description

/**
 *
 */
abstract class ValueIdReplacement<DataT> {

    // Not available until training complete
    lateinit var accessor: AccessData
}

class AccessData

class TransformDescriptionReplacement {
    val declaredIds: List<ValueIdReplacement>
}
