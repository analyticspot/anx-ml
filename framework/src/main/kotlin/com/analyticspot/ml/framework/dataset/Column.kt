package com.analyticspot.ml.framework.dataset

import java.util.stream.Stream

/**
 * The interface for all the columns in a [DataSet]. Different subtypes may use different storage.
 */
interface Column<out T : Any?> : Iterable<T> {
    val size: Int
    operator fun get(rowIndex: Int): T

    /**
     * Returns the values in the column as a `Sequence`
     */
    fun sequence(): Sequence<T> = this.sequence()

    /**
     * Returns the values in the column as a Java 8 `Stream`.
     */
    fun stream(): Stream<out T> = this.stream()

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
class ListColumn<T>(private val data: List<T>) : Column<T> {
    override val size: Int
        get() = data.size

    override fun get(rowIndex: Int): T {
        return data[rowIndex]
    }

    override fun iterator(): Iterator<T> = data.iterator()
}
