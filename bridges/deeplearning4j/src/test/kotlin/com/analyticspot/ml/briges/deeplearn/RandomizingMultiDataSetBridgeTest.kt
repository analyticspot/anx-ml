package com.analyticspot.ml.briges.deeplearn

import com.analyticspot.ml.framework.dataset.DataSet
import com.analyticspot.ml.framework.description.ColumnId
import org.assertj.core.api.Assertions.assertThat
import org.datavec.api.conf.Configuration
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.slf4j.LoggerFactory
import org.testng.annotations.Test
import java.util.Random

class RandomizingMultiDataSetBridgeTest {
    companion object {
        private val log = LoggerFactory.getLogger(RandomizingMultiDataSetBridgeTest::class.java)

        val SAMPLE_DATA_RESOURCE = "/DataSetIteratorData.csv"
    }
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
        // Intentionally setting a batch size that doesn't evenly divide into the number of rows so we can test the
        // final, partial batch.
        val batchSize = 12

        // allData is a list of columns, each of type Double (we'll convert the targets to int later)
        val allData: MutableList<MutableList<Double>> = mutableListOf()
        for (i in 0 until totalColumns) {
            allData.add(mutableListOf())
        }

        this.javaClass.getResourceAsStream(SAMPLE_DATA_RESOURCE).bufferedReader().useLines { lineSequence ->
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

        val dl4jReader = CSVRecordReader(0, ",")
        dl4jReader.initialize(FileSplit(ClassPathResource(SAMPLE_DATA_RESOURCE).file))
        val dl4jIter = RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("fazzle", dl4jReader)
                .addInput("fazzle", 0, 1)
                .addInput("fazzle", 2, 3)
                .addOutputOneHot("fazzle", 4, 4)
                .addOutputOneHot("fazzle", 5, 4)
                .build()

        while (ourIter.hasNext()) {
            log.debug("Checking a batch")
            assertThat(dl4jIter.hasNext())
            val ourMds = ourIter.next()
            val dl4jMds = dl4jIter.next()

            val ourFeatures = ourMds.features
            val dl4jFeatures = dl4jMds.features
            assertThat(ourFeatures).hasSameSizeAs(dl4jFeatures)
            for (fidx in ourFeatures.indices) {
                assertThat(ourFeatures[fidx].shape()).isEqualTo(dl4jFeatures[fidx].shape())
            }

            val ourTargets = ourMds.labels
            val dl4jTargets = dl4jMds.labels
            assertThat(ourTargets).hasSameSizeAs(dl4jTargets)
            for (tidx in ourTargets.indices) {
                assertThat(ourTargets[tidx].shape()).isEqualTo(dl4jTargets[tidx].shape())
            }
        }
    }
}
