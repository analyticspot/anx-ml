/*
 * Copyright (C) 2017 Analytic Spot.
 * 
 * This file is part of the ANX ML library.
 * 
 * The ANX ML library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * The ANX ML library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with the ANX ML libarary.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

package com.analyticspot.ml.framework.testutils

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.dataset.ListColumn
import com.analyticspot.ml.framework.datatransform.LearningTransform
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.ColumnIdGroup
import com.fasterxml.jackson.databind.annotation.JsonDeserialize
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder
import org.slf4j.LoggerFactory
import java.util.TreeMap
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * An example of an unsupervised learner that doesn't know in advance how many outputs it will have. This one takes in
 * [DataSet] instances that contain a `List<String>` and assigns each unique string to a output. The value of that
 * output is the number of times that string occurred. This is very much like a bag-of-words representation.
 *
 * @param sourceColumn a [ColumnId] to retrieve the list of words from the input.
 * @param resultPrefix the prefix for the [ColumnIdGroup] that this will produce. The resulting columns
 * will be `prefix-X` where `X` is one of the words in the list.
 * @param wordSet the set of words in the learned vocabulary. This is filled in during training and then, in [tranform]
 * we ensure that there is one column for every word in this set and only words in this set are counted.
 */
@JsonDeserialize(builder = WordCounts.Builder::class)
class WordCounts private constructor(
        val sourceColumn: ColumnId<List<String>>,
        val resultId: ColumnIdGroup<Int>,
        val wordSet: MutableSet<String>) : LearningTransform {

    constructor(sourceColumn: ColumnId<List<String>>, resultId: ColumnIdGroup<Int>)
            : this(sourceColumn, resultId, mutableSetOf())

    companion object {
        private val log = LoggerFactory.getLogger(WordCounts::class.java)
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val counts: MutableMap<String, MutableList<Int>> = TreeMap()
        wordSet.forEach { counts.put(it, mutableListOf()) }
        dataSet.column(sourceColumn).forEach { transform(it, counts) }

        val resultData = DataSet.build {
            counts.forEach {
                addColumn(resultId.generateId(it.key), ListColumn(it.value))
            }
        }

        return CompletableFuture.completedFuture(resultData)
    }

    fun transform(words: List<String>?, counts: MutableMap<String, MutableList<Int>>) {
        val localCounts = mutableMapOf<String, Int>()
        // Fill in localcounts so it's a map from word to # of times that word was seen in this data set.
        words?.forEach {
            if (it in words) {
                localCounts[it] = localCounts[it]?.plus(1) ?: 1
            }
        }
        // Then, for each word in the vocabulary add an entry in counts a map from columns to the values for those
        // columns
        wordSet.forEach {
            val columnValue = localCounts[it] ?: 0
            counts[it]!!.add(columnValue)
        }

    }

    override fun trainTransform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        // Training phase
        dataSet.column(sourceColumn).forEach {
            it?.forEach { wordSet.add(it) }
        }

        return transform(dataSet, exec)
    }

    // This should be unnecessary but is a workaround for this bug:
    //
    // https://github.com/FasterXML/jackson-databind/issues/1489
    //
    // We should be able to remove this when that gets fixed.
    @JsonPOJOBuilder(withPrefix = "set")
    class Builder() {
        lateinit var sourceColumn: ColumnId<List<String>>
        lateinit var resultId: ColumnIdGroup<Int>
        lateinit var wordSet: MutableSet<String>

        fun build(): WordCounts = WordCounts(sourceColumn, resultId, wordSet)
    }
}
