package com.analyticspot.ml.framework.datagraph;

import static org.assertj.core.api.Assertions.assertThat;

import com.analyticspot.ml.framework.dataset.DataSet;
import com.analyticspot.ml.framework.description.ColumnId;
import com.analyticspot.ml.framework.testutils.LowerCaseTransform;
import com.analyticspot.ml.framework.testutils.TrueIfSeenTransform;
import org.assertj.core.util.Lists;
import org.testng.annotations.Test;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * This is testing Java interoperability. Most of the real tests are in Kotlin.
 */
public class GraphExectionJavaTest {
  @Test
  public void testCanCreateAndExecuteAGraph() throws Exception {
    DataGraph.GraphBuilder graphBuilder = DataGraph.builder();

    ColumnId<String> wordsId = new ColumnId<>("words", String.class);
    ColumnId<Boolean> targetId = new ColumnId<>("target", Boolean.class);
    ColumnId<Boolean> predictionId = new ColumnId<>("prediction", Boolean.class);

    GraphNode source = graphBuilder.source()
        .withValue(wordsId)
        .withTrainOnlyValue(targetId)
        .build();

    GraphNode toLower = graphBuilder.addTransform(source,
        new LowerCaseTransform());

    GraphNode trueIfSeen = graphBuilder.addTransform(toLower, source,
        new TrueIfSeenTransform(wordsId, targetId, predictionId));

    graphBuilder.setResult(trueIfSeen);

    DataGraph graph = graphBuilder.build();

    DataSet trainingData = graph.createTrainingSource(new Object[][]{
            {"foO", true},
            {"bar", false},
            {"BAZ", true}
        });

    DataSet trainResult = graph.trainTransform(
        trainingData, Executors.newFixedThreadPool(4)).get();

    List<Boolean> trainPredictions = trainResult.column(predictionId).stream().collect(Collectors.toList());

    assertThat(trainPredictions).isEqualTo(Lists.newArrayList(true, false, true));

    DataSet testData = graph.createSource(new Object[][]{
        { "FOO" }, { "bip" }, { "baz" }, { "blah" }
    });

    DataSet testResult = graph.transform(testData, Executors.newFixedThreadPool(4)).get();
    List<Boolean> testPredictions = testResult.column(predictionId).stream().collect(Collectors.toList());
    assertThat(testPredictions).isEqualTo(Lists.newArrayList(true, false, true, false));
  }
}
