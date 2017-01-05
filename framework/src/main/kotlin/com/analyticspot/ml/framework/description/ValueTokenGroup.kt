package com.analyticspot.ml.framework.description

/**
 * A [ValueIdGroup] is the id for a group of tokens. This allows you to map that to the actual tokens. Note that some
 * of the functions here will throw an exception if they're called before the underlying data transform has been trained
 * and the tokens are known.
 */
interface ValueTokenGroup<DataT> {
    /**
     * The id for this [ValueTokenGroup]
     */
    val id: ValueIdGroup<DataT>

    /**
     * The prefix for all the tokens in the group. Tokens have names that are `prefix`, a `-` character, and then some
     * character string that uniquely identifies the token.
     */
    val prefix: String
        get() = id.prefix

    /**
     * The data type for all [ValueToken] instances in [tokens].
     */
    val clazz: Class<DataT>
        get() = id.clazz

    /**
     * Returns the number of tokens in this group. Generally not available until producing transform has been trained.
     */
    fun numTokens(): Int

    /**
     * Returns the list of tokens in [tokens] in this group. Generally not available until the producing transform has
     * been trained.
     */
    fun tokens(): List<ValueToken<DataT>>

    /**
     * Returns the tokens in [tokens] as a `Set`. Generally not available until the producing transform has been
     * trained.
     */
    fun tokenSet(): Set<ValueToken<DataT>>
}
