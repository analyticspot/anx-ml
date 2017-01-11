package com.analyticspot.ml.framework.description

/**
 * A [ValueIdGroup] that consists of multiple other [ValueId] and [ValueIdGroup].
 */
class AggregateValueIdGroup<ValueT>(builder: Builder<ValueT>) : ValueIdGroup<ValueT>(builder.name, builder.clazz) {
    internal val valueIds: List<ValueId<ValueT>> = builder.valueIds
    internal val valueIdGroups: List<ValueIdGroup<ValueT>> = builder.valueIdGroups

    class Builder<ValueT>(val name: String, val clazz: Class<ValueT>) {
        val valueIds = mutableListOf<ValueId<ValueT>>()
        val valueIdGroups = mutableListOf<ValueIdGroup<ValueT>>()
    }
}
