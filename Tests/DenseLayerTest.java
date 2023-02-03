package Tests;
import static org.junit.Assert.assertArrayEquals;

import java.util.stream.Stream;

import org.junit.Test;

import src.DenseLayer;
import src.Main;
import src.Matrix;

public class DenseLayerTest {
    @Test
    public void forward(){
        DenseLayer dense = new DenseLayer(4, 5);
        dense.weights.testSeed();
        dense.biases.testSeed();
        Matrix inputs = new Matrix(5, 1);
        inputs.matrix = new Double[]{1.0, -2.0, 3.0, -4.0, 5.0};
        Matrix denseOutput = dense.forward(inputs);
        Matrix expectedMatrix = new Matrix(4, 1);
        expectedMatrix.matrix = new Double[]{1.2, 2.8, 4.4, 6.0};
        double[] actual = Stream.of(denseOutput.matrix).mapToDouble(Double::doubleValue).toArray();
        double[] expected = Stream.of(expectedMatrix.matrix).mapToDouble(Double::doubleValue).toArray();
        assertArrayEquals(expected, actual, 0.001);
    }
    @Test 
    public void backward(){
        DenseLayer dense = new DenseLayer(1, 3);
        dense.weights.matrix = new Double[]{-3.0, -1.0, 2.0};
        dense.biases.matrix = new Double[]{1.0};
        Matrix input = new Matrix(3, 1);
        input.matrix = new Double[]{1.0, -2.0, 3.0};
        dense.forward(input);
        Matrix dvalues = new Matrix(1, 1);
        dvalues.matrix = new Double[]{1.0};
        Matrix actualDerivativeMatrix = dense.backwards(dvalues);
        double[] expectedWeights = new double[]{-3 - 1 * Main.ALPHA, -1 - -2 * Main.ALPHA, 2 - 3 * Main.ALPHA};
        double[] expectedBiases = new double[]{1 - 1 * Main.ALPHA};
        double[] expectedDerivatives = new double[]{-3.0, -1.0, 2.0};
        double[] actualWeights = Stream.of(dense.weights.matrix).mapToDouble(Double::doubleValue).toArray();
        assertArrayEquals(expectedWeights, actualWeights, 0.000001);
        double[] actualBiases = Stream.of(dense.biases.matrix).mapToDouble(Double::doubleValue).toArray();
        assertArrayEquals(expectedBiases, actualBiases, 0.000001);
        double[] actualDerivatives = Stream.of(actualDerivativeMatrix.matrix).mapToDouble(Double::doubleValue).toArray();
        assertArrayEquals(actualDerivatives, expectedDerivatives, 0.000001);
    }
}