package com.analyticspot.ml.framework.datatransform

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import com.analyticspot.ml.framework.metadata.CategoricalFeatureMetaData
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService

/**
 * A [DataTransform] that converts the requested columns to categorical features. Specifically it:
 *
 * * Creates [CategoricalFeatureMetaData] conforming to the observed values for each column when [trainTransform] is
 *   called.
 * * When [transform] is called it coerces the column to conform to the metadata by replacing previously unknown values
 *   with `null` (using [CategicalFeatureMetaData.makeColumnConformToMetaData]. It also attaches the metadata to the
 *   [DataSet] again.
 *
 * Thus, the input to this transform is always a data set without metadata for the requested columns and the output
 * is always a data set with metadata plus the columns may be modified during [transform] if they don't conform to the
 * meta data. The output [DataSet] will contain **only** the columns passed to the constructor.
 */
class StringToCategoricalTransform(val columnsToConvert: List<ColumnId<String>>) : LearningTransform {
    lateinit var colNameToMetaData: Map<String, CategoricalFeatureMetaData>

    constructor(vararg columnsToConvert: ColumnId<String>) : this(columnsToConvert.toList())

    /**
     * Used by Jackson for deserialization.
     */
    @JsonCreator
    constructor(@JsonProperty("columnsToConvert") columnsToConvert: List<ColumnId<String>>,
            @JsonProperty("colNameToMetaData") colNameToMetaData: Map<String, CategoricalFeatureMetaData>)
        : this(columnsToConvert) {
        this.colNameToMetaData = colNameToMetaData
    }

    override fun transform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val resultBldr = DataSet.builder()

        columnsToConvert.forEach {
            val column = dataSet.column(it)
            val metaData = colNameToMetaData[it.name]!!
            val convertedColumn = metaData.makeColumnConformToMetaData(column)
            resultBldr.addColumn(it, convertedColumn, metaData)
        }
        return CompletableFuture.completedFuture(resultBldr.build())
    }

    override fun trainTransform(dataSet: DataSet, exec: ExecutorService): CompletableFuture<DataSet> {
        val resultBldr = DataSet.builder()
        val metaMap = mutableMapOf<String, CategoricalFeatureMetaData>()
        columnsToConvert.forEach {
            val column = dataSet.column(it)
            // We have to allow missing as the transform step can create data with missing values
            val metaData = CategoricalFeatureMetaData.fromStringColumn(column).copyWithMissingAllowed()
            metaMap[it.name] = metaData
            resultBldr.addColumn(it, column, metaData)
        }
        colNameToMetaData = metaMap
        return CompletableFuture.completedFuture(resultBldr.build())
    }
}
