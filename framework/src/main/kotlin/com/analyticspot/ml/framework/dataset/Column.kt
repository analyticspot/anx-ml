package com.analyticspot.ml.framework.dataset

/**
 * The interface for all the columns in a [DataSet]. Different subtypes may use different storage.
 */
interface Column<out T : Any?> {
    val size: Int
    operator fun get(rowIndex: Int): T

    fun <R> map(mapFun: (T) -> R): Column<R>
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

    override fun <R> map(mapFun: (T) -> R): Column<R> {
        val resultList = data.map { mapFun(it) }
        return ListColumn(resultList)
    }
}
