package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.testng.annotations.Test

class RandomizingMultiDataSetBridgeTest {
    // DeepLearning4j supplies several MultiDataSetIterator implementations (that all read from files and don't work
    // with in-memory data and don't work with our DataSet instances). We want to test that we produce the same exact
    // results as they do on the same data. So we have a CSV file we can use with DL4j and we then manually read that
    // CSV file into a DataSet and build our own MutliDataSetIterator and ensure they all produce the same stuff. The
    // input file contains 6 columns. The first two are double and will be one DataSet in our MulitDataSet, the 2nd 2
    // are a double and an integer. The 5th and 6th are integers that represent the target and will be one-hot encoded.
    @Test
    fun testProducesSameResultAsDl4j() {
        val totalColumns = 6
        val numRows = 20
        val batchSize = 10

        // allData is a list of columns, each of type Double (we'll convert the targets to int later)
        val allData: MutableList<MutableList<Double>> = mutableListOf()
        for (i in 0 until totalColumns) {
            allData.add(mutableListOf())
        }

        this.javaClass.getResourceAsStream("/DataSetIteratorData.csv").bufferedReader().useLines { lineSequence ->
            lineSequence.forEach {
                val parts = it.split(",")
                check(parts.size == 6)
                parts.forEachIndexed { colIdx, part ->
                    allData[colIdx].add(part.toDouble())
                }
            }
        }

        val ds1 = DataSet.build {
            addColumn(ColumnId.create<Double>("c11"), allData[0])
            addColumn(ColumnId.create<Double>("c12"), allData[1])
        }

        val ds2 = DataSet.build {
            addColumn(ColumnId.create<Double>("c21"), allData[2])
            addColumn(ColumnId.create<Double>("c22"), allData[3])
        }

        val dsTargets = DataSet.build {
            addColumn(ColumnId.create<Int>("t1"), allData[4].map { it.toInt() })
            addColumn(ColumnId.create<Int>("t2"), allData[5].map { it.toInt() })
        }

        check(ds1.numRows == numRows)

        val ourIter = RandomizingMultiDataSetBridge(batchSize, listOf(ds1, ds2), dsTargets)

        while (ourIter.hasNext()) {
            val mds = ourIter.next()
            assertThat(mds.features).hasSize(2)
            assertThat(mds.labels).hasSize(2)
        }
    }
}
