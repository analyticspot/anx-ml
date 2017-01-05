package com.analyticspot.ml.framework.description

/**
 * The [ValueTokenGroup] analog of [IndirectValueToken].
 *
 * @param obsIndex the index of the source in the array of [Observation]/[DataSet] that makes up the composite
 *     [Observartion]/[DataSet].
 * @param sourceGroup the [ValueTokenGroup] that is the source for this.
 */
class IndirectValueTokenGroup<DataT>(private val obsIndex: Int,
        private val sourceGroup: ValueTokenGroup<DataT>) : ValueTokenGroup<DataT> {
    override val id: ValueIdGroup<DataT>
        get() = sourceGroup.id

    private val theTokens by lazy {
        sourceGroup.tokens().map { IndirectValueToken(obsIndex, it) }
    }

    private val theTokenSet by lazy {
        theTokens.toSet()
    }

    override fun numTokens(): Int = sourceGroup.numTokens()

    override fun tokens(): List<ValueToken<DataT>> = theTokens

    override fun tokenSet(): Set<ValueToken<DataT>> = theTokenSet
}
