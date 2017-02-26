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

package com.analyticspot.ml.framework.dataset

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.annotation.JsonProperty.Access
import com.fasterxml.jackson.annotation.JsonTypeId
import com.fasterxml.jackson.annotation.JsonTypeInfo
import com.fasterxml.jackson.annotation.JsonTypeInfo.As
import com.fasterxml.jackson.annotation.JsonTypeInfo.Id
import kotlinx.support.jdk8.collections.spliterator
import java.util.stream.Stream
import java.util.stream.StreamSupport

/**
 * The interface for all the columns in a [DataSet]. Different subtypes may use different storage.
 */
@JsonTypeInfo(use= Id.CLASS, include= As.PROPERTY, property="class")
interface Column<out T : Any?> : Iterable<T> {
    @get:JsonIgnore
    val size: Int

    operator fun get(rowIndex: Int): T

    /**
     * Returns the values in the column as a `Sequence`
     */
    fun sequence(): Sequence<T> = this.sequence()

    /**
     * Returns the values in the column as a Java 8 `Stream`.
     */
    fun stream(): Stream<out T> = StreamSupport.stream(this.spliterator(), false)

    /**
     * Maps one column to a new column via a function.
     */
    fun <R> mapToColumn(transform: (T) -> R): Column<R> {
        return ListColumn(this.map { transform(it) })
    }
}

/**
 * A [Column] that stores its data in an `Array`.
 */
class ListColumn<T> @JsonCreator constructor(
        @JsonProperty("data", access = Access.READ_WRITE) private val data: List<T>) : Column<T> {
    override val size: Int
        get() = data.size

    override fun get(rowIndex: Int): T {
        return data[rowIndex]
    }

    override fun iterator(): Iterator<T> = data.iterator()
}
