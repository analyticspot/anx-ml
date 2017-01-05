package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.IterableDataSet
import com.analyticspot.ml.framework.datatransform.LearningTransform
import com.analyticspot.ml.framework.datatransform.TransformDescription
import com.analyticspot.ml.framework.description.IndexValueToken
import com.analyticspot.ml.framework.description.ValueId
import com.analyticspot.ml.framework.description.ValueIdGroup
import com.analyticspot.ml.framework.description.ValueToken
import com.analyticspot.ml.framework.description.ValueTokenGroupFromList
import com.analyticspot.ml.framework.observation.ArrayObservation
import com.analyticspot.ml.framework.observation.Observation
import org.slf4j.LoggerFactory
import java.util.concurrent.CompletableFuture

/**
 * An example of an unsupervised learner that doesn't know in advance how many outputs it will have. This one takes in
 * [Observation] instances that contain a `List<String>` and assigns each unique string to a output. The value of that
 * output is the number of times that string occurred. This is very much like a bag-of-words representation.
 *
 * @param sourceToken a [ValueToken] to retrieve the list of words from the input.
 * @param resultPrefix the prefix for the [ValueIdGroup]/[ValueTokenGroup] that this will produce. The resulting tokens
 * will be `prefix-X` where `X` is one of the words in the list.
 */
class WordCounts(private val sourceToken: ValueToken<List<String>>, resultId: ValueIdGroup<Int>) : LearningTransform {
    // Tells you the index for each word.
    private val wordMap = mutableMapOf<String, Int>()
    private val tokenGroupAndSetter = ValueTokenGroupFromList.create(resultId)

    override val description = TransformDescription(listOf(), tokenGroups = listOf(tokenGroupAndSetter.tokenGroup))

    companion object {
        private val log = LoggerFactory.getLogger(WordCounts::class.java)
    }

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val obsList = dataSet.map { transform(it) }
        return CompletableFuture.completedFuture(IterableDataSet(obsList))
    }

    fun transform(obs: Observation): Observation {
        val result = IntArray(wordMap.size)
        val words = obs.value(sourceToken)
        words.forEach {
            val idx = wordMap[it]
            if (idx == null) {
                log.debug("$it is not in the vocabulary. Ignored.")
            } else {
                result[idx] += 1
            }
        }
        // TODO: Converting from IntArray to Array<Int> is not very efficient as the latter is really a Java
        // Array<Integer> rather than an int[]. We should create Observation subclasses for primitive arrays.
        return ArrayObservation(result.toTypedArray())
    }

    override fun trainTransform(dataSet: DataSet): CompletableFuture<DataSet> {
        // Training phase
        dataSet.forEach { observation ->
            val words = observation.value(sourceToken)
            for (word in words) {
                if (!wordMap.containsKey(word)) {
                    wordMap[word] = wordMap.size
                }
            }
        }

        // Now that we know all the tokens, set them into the ValueTokenGroup
        val tokens = wordMap.map {
            val tokenName = "${tokenGroupAndSetter.tokenGroup.prefix}${ValueId.GROUP_SEPARATOR}${it.key}"
            IndexValueToken(it.value, ValueId.create<Int>(tokenName))
        }

        tokenGroupAndSetter.setter(tokens)
        return transform(dataSet)
    }
}
