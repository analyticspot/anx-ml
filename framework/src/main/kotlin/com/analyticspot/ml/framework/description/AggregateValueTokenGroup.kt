package com.analyticspot.ml.framework.description

import com.analyticspot.ml.framework.datagraph.GraphNode

/**
 * The [ValueTokenGroup] for an [AggregateValueIdGroup].
 */
internal class AggregateValueTokenGroup<DataT>(
        override val id: AggregateValueIdGroup<DataT>, source: GraphNode) : ValueTokenGroup<DataT> {
    private val srcTokens: List<ValueToken<DataT>> = id.valueIds.map { source.token(it) }
    private val srcTokenGroups: List<ValueTokenGroup<DataT>> = id.valueIdGroups.map { source.tokenGroup(it) }

    private val _numTokens by lazy {
        srcTokens.size + srcTokenGroups.map { it.numTokens() }.sum()
    }

    private val _tokens: List<ValueToken<DataT>> by lazy {
        srcTokenGroups.flatMap { it.tokens() }.plus(srcTokens)
    }

    private val _tokenSet: Set<ValueToken<DataT>> by lazy {
        _tokens.toSet()
    }

    override fun numTokens(): Int = _numTokens

    override fun tokens(): List<ValueToken<DataT>> = _tokens

    override fun tokenSet(): Set<ValueToken<DataT>> = _tokenSet
}
