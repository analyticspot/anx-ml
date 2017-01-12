package com.analyticspot.ml.framework.description

/**
 * The [ValueTokenGroup] for an [AggregateValueIdGroup].
 */
internal class AggregateValueTokenGroup<DataT>(
        override val id: ValueIdGroup<DataT>,
        private val srcTokens: List<ValueToken<DataT>>,
        private val srcTokenGroups: List<ValueTokenGroup<DataT>>) : ValueTokenGroup<DataT> {

    override val declaredTokens: List<ValueToken<DataT>>
        get() = srcTokens

    constructor(id: AggregateValueIdGroup<DataT>, source: TransformDescription) :
            this(id, id.valueIds.map { source.token(it) }, id.valueIdGroups.map { source.tokenGroup(it) })

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
}
