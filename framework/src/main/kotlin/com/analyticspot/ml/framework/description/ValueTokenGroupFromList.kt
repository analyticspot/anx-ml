package com.analyticspot.ml.framework.description

import java.util.concurrent.atomic.AtomicReference

/**
 * A [ValueTokenGroup] that allows the caller/creator to set the tokens, when they are known, by calling a function
 * once the tokens are known. Typically this is done at the end of the training phase by the transform that generated
 * the data and the tokens.
 */
class ValueTokenGroupFromList<DataT> private constructor(override val id: ValueIdGroup<DataT>)
    : ValueTokenGroup<DataT> {
    // HERE IS PROBLEM! Tokens are not yet available but some have been declared! This class doesn't know what type
    // of tokens we'll end up with so it doesn't know how to create the right thing.
    override val declaredTokens: List<ValueToken<DataT>> = listOf()
    private var theTokens: AtomicReference<List<ValueToken<DataT>>?> = AtomicReference()

    companion object {
        private val NOT_READY_ERR = IllegalStateException("You can't retrieve tokens before training is complete.")

        /**
         * This is how you create the [ValueTokenGroup]. The returned object contains both the token group and a
         * function you can call when the set of tokens is known.
         */
        fun <T> create(id: ValueIdGroup<T>): TokenGroupAndSetter<T> {
            val group = ValueTokenGroupFromList(id)
            return TokenGroupAndSetter(group, { toks -> group.setTokens(toks) })
        }
    }

    override fun numTokens(): Int {
        return theTokens.get()?.size ?: throw NOT_READY_ERR
    }

    override fun tokens(): List<ValueToken<DataT>> {
        return theTokens.get() ?: throw NOT_READY_ERR
    }

    // This is private for 2 reasons:
    //
    // (1) we don't want the consuming transforms to be able to modify the tokens
    // (2) we want to keep the implementation of the setting transforms as independent of this as we can.
    //
    // Thus, the transform that creates the token group via the static factory gets a function they can use when the
    // data is available. That way only the creator can modify the tokens and the creator need not know anything about
    // how that actually happens
    private fun setTokens(toks: List<ValueToken<DataT>>) {
        theTokens.set(toks)
    }

    data class TokenGroupAndSetter<DataT>(
            val tokenGroup: ValueTokenGroupFromList<DataT>,
            val setter: (List<ValueToken<DataT>>) -> Unit)
}
