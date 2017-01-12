package com.analyticspot.ml.framework.description

/**
 * Utility functions for working with [ValueToken] and [ValueTokenGroup].
 */

/**
 * Returns a [ValueTokenGroup] with just a single token. The resulting group will be named `groupName`.
 */
fun <T> groupFromToken(groupName: String, tok: ValueToken<T>): ValueTokenGroup<T> {
    val groupId = ValueIdGroup(groupName, tok.clazz)
    return AggregateValueTokenGroup(groupId, listOf(tok), listOf())
}
