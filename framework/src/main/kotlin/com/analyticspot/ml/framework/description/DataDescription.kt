package com.analyticspot.ml.framework.description

import java.util.SortedMap

/**
 * A description of a [DataSet]. This doesn't contain any data but it does know how to generate data of the given type,
 * including knowing all the other [DataSet] and [DataTransform] inputs required to realize this data set. Thus, a
 * single [DataDescription] is sufficient to reproduce the entire data graph that terminates with this description.
 *
 * @param tokens indicates which values are available in this data set during transformation.
 * @param trainOnlyTokens tokens that are only available in train or trainTransform mode. Once trained these tokens are
 *    no longer available. A good example of such tokens would be a token for the target value.
 * @param tokenGroups tokens which have been grouped. See [TokenGroup] for more information.
 */
open class DataDescription(builder: Builder) {
    val tokens: List<ValueToken<*>>
    val trainOnlyTokens: List<ValueToken<*>>
    val tokenGroups: List<TokenGroup<*>>

    private val tokenMap: SortedMap<String, ValueToken<*>> = sortedMapOf()
    private val tokenGroupMap: SortedMap<String, TokenGroup<*>> = sortedMapOf()

    init {
        tokens = builder.tokens
        trainOnlyTokens = builder.trainOnlyTokens
        tokenGroups = builder.tokenGroups
        tokens.asSequence().plus(trainOnlyTokens).forEach {
            check(!tokenMap.containsKey(it.name)) {
                "A token with name ${it.name} is already present in this data set."
            }
            tokenMap.put(it.name, it)
        }

        tokenGroups.forEach {
            check(!tokenGroupMap.containsKey(it.prefix)) {
                "A token group with prefix ${it.prefix} is already present in this data set."
            }
            tokenGroupMap.put(it.prefix, it)
        }

    }

    companion object {
        fun build(init: Builder.() -> Unit): DataDescription {
            return with(Builder()) {
                init()
                return build()
            }
        }
    }

    fun <T> token(name: String, clazz: Class<T>): ValueToken<T> {
        val token = tokenMap[name] ?: throw IllegalArgumentException("Token $name not found")
        if (token.clazz == clazz) {
            @Suppress("UNCHECKED_CAST")
            return token as ValueToken<T>
        } else {
            throw IllegalArgumentException("Token $name is not of type $clazz")
        }
    }

    inline fun <reified T : Any> token(name: String): ValueToken<T> {
        return token(name, T::class.java)
    }

    fun <T> tokenGroup(prefix: String, clazz: Class<T>): TokenGroup<T> {
        val tokenGroup = tokenGroupMap[prefix]
                ?: throw IllegalArgumentException("TokenGroup with prefix $prefix not found")
        if (tokenGroup.clazz == clazz) {
            @Suppress("UNCHECKED_CAST")
            return tokenGroup as TokenGroup<T>
        } else {
            throw IllegalArgumentException("TokenGroup with prefix $prefix is not of type $clazz")
        }
    }

    inline fun <reified T : Any> tokenGroup(name: String): TokenGroup<T> {
        return tokenGroup(name, T::class.java)
    }

    open class Builder {
        val tokens: MutableList<ValueToken<*>> = mutableListOf()
        val trainOnlyTokens: MutableList<ValueToken<*>> = mutableListOf()
        val tokenGroups: MutableList<TokenGroup<*>> = mutableListOf()

        open fun build(): DataDescription {
            return DataDescription(this)
        }
    }
}
