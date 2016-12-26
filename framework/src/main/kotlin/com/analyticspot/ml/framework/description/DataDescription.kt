package com.analyticspot.ml.framework.description

import org.slf4j.LoggerFactory
import java.util.SortedMap

/**
 * A description of a [DataSet]. This doesn't contain any data but it does know how to generate data of the given type,
 * including knowing all the other [DataSet] and [DataTransform] inputs required to realize this data set. Thus, a
 * single [DataDescription] is sufficient to reproduce the entire data graph that terminates with this description.
 *
 * @param tokens indicates which values are available in this data set during transformation.
 * @param trainOnlyTokens tokens that are only available in train or trainTransform mode. Once trained these tokens are
 *    no longer available. A good example of such tokens would be a token for the target value.
 * @param tokenGroups tokens which have been grouped. See [ValueIdGroup] for more information.
 */
open class DataDescription(builder: Builder) {
    val tokens: List<ValueToken<*>>
    val trainOnlyTokens: List<ValueToken<*>>
    val tokenGroups: List<ValueIdGroup<*>>

    private val tokenMap: SortedMap<ValueId<*>, ValueToken<*>> = sortedMapOf()

    init {
        log.debug("DataDescription being constructed with {} tokens and {} token groups",
                builder.tokens.size, builder.tokenGroups.size)
        tokens = builder.tokens
        trainOnlyTokens = builder.trainOnlyTokens
        tokenGroups = builder.tokenGroups
        tokens.asSequence().plus(trainOnlyTokens).forEach {
            log.debug("Adding token named {} to the token map", it.name)
            check(!tokenMap.containsKey(it.id)) {
                "A token with name ${it.name} is already present in this data set."
            }
            tokenMap.put(it.id, it)
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(DataDescription::class.java)
    }

    fun <T> token(valId: ValueId<T>): ValueToken<T> {
        val tok = tokenMap[valId] ?: throw IllegalArgumentException("Token ${valId.name} not found")
        if (tok.clazz == valId.clazz) {
            @Suppress("UNCHECKED_CAST")
            return tok as ValueToken<T>
        } else {
            throw IllegalArgumentException("Token ${valId.name} is not of type ${valId.clazz}")
        }
    }

    open class Builder {
        val tokens: MutableList<ValueToken<*>> = mutableListOf()
        val trainOnlyTokens: MutableList<ValueToken<*>> = mutableListOf()
        val tokenGroups: MutableList<ValueIdGroup<*>> = mutableListOf()

        open fun build(): DataDescription {
            return DataDescription(this)
        }
    }
}
