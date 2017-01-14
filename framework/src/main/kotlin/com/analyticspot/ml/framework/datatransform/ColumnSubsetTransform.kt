package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.description.TransformDescription
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.annotation.JsonProperty.Access
import java.util.concurrent.CompletableFuture

/**
 * Takes in a source [DataSet] and returns a new [DataSet] that contains a subset of the columns in the source. There
 * is also an option to rename the columns that are retained.
 */
class ColumnSubsetTransform : SingleDataTransform {
    override val description: TransformDescription

    @JsonProperty(access = Access.READ_ONLY)
    val keepMap: Map<ColumnId<*>, ColumnId<*>>

    @JsonCreator
    private constructor(@JsonProperty("keepMap") keepMap: Map<ColumnId<*>, ColumnId<*>>) {
        this.keepMap = keepMap
        description = TransformDescription(keepMap.values.toList())
    }

    private constructor(builder: Builder) : this(builder.keepMap)

    companion object {
        fun builder(): Builder = Builder()

        fun build(init: Builder.() -> Unit): ColumnSubsetTransform {
            return with(builder()) {
                init()
                build()
            }
        }
    }

    override fun transform(dataSet: DataSet): CompletableFuture<DataSet> {
        val bldr = DataSet.builder()
        keepMap.forEach { entry ->
            @Suppress("UNCHECKED_CAST")
            bldr.addColumn(entry.value as ColumnId<Any>, dataSet.column(entry.key))
        }
        return CompletableFuture.completedFuture(bldr.build())
    }

    class Builder {
        // Maps from the id of a column to keep to a new id for that column.
        internal val keepMap = mutableMapOf< ColumnId<*>, ColumnId<*>>()

        /**
         * The result should contain the given column.
         */
        fun keep(id: ColumnId<*>): Builder {
            require(id !in keepMap) {
                "Column $id already specified"
            }
            keepMap[id] = id
            return this
        }

        /**
         * The result should contain `srcCol` but in the output that data will have name `newName`.
         */
        fun keepAndRename(srcCol: ColumnId<*>, newName: String): Builder {
            require(srcCol !in keepMap) {
                "Column $srcCol already specified"
            }
            keepMap[srcCol] = ColumnId(newName, srcCol.clazz)
            return this
        }

        /**
         * The result should contain `srcCol` but in the output that data will have id `newId`.
         */
        fun <T : Any> keepAndRename(srcCol: ColumnId<T>, newId: ColumnId<out T>): Builder {
            require(srcCol !in keepMap) {
                "Column $srcCol already specified"
            }
            keepMap[srcCol] = newId
            return this
        }

        fun build(): ColumnSubsetTransform {
            return ColumnSubsetTransform(this)
        }
    }
}
