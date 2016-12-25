package com.analyticspot.ml.framework.description

/**
 * Some algorithms produce an unknown number of values. For example, a "bag of words" transform will produce one `Int`
 * value for each unique word in the training corpus: until it's been trained we can't know many words there will be.
 * Still, we often want to tell consuming [DataTransformer] which values to consume from a [DataSet] so we need a way to
 * refer to a group of tokens. This is done via a [TokenGroup]: it refers to an unknown number of [ValueToken] that
 * all have the same type and whose names all have the same prefix.
 */
data class TokenGroup<T>(val prefix: String, val clazz: Class<T>) {
    companion object {
        inline fun <reified T : Any> create(prefix: String): TokenGroup<T> {
            return TokenGroup(prefix, T::class.java)
        }
    }
}
