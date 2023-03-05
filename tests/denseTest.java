package tests;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

import src.DenseLayer;
import src.Main;
import src.Matrix;
import src.Utils;

public class denseTest {
    @Test
    public void test() {
        Main.ALPHA = 0.001;
        Main.BATCHSIZE = 1;
        DenseLayer dense = new DenseLayer(3);
        Matrix forwardInput = new Matrix(1, 1, 2);
        Matrix backwardInput = new Matrix(1, 3, 1);
        dense.weights = new Matrix(1, 2, 3);
        dense.biases = new Matrix(1, 3, 1);
        dense.outputs = new Matrix(1, 3, 1);
        dense.weights.matrix = new Double[][]{{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}};
        dense.biases.matrix = new Double[][]{{0.0, 0.0, 0.0}};
        dense.initialized = true;
        forwardInput.matrix = new Double[][]{{69.0, 420.0}};
        backwardInput.matrix = new Double[][]{{2.0, -69.0, -10.0}};
        Matrix theResult = dense.forward(forwardInput);
        Double[] expectedForwardResults = new Double[]{90.9, 188.7, 286.5};
        assertArrayEquals(expectedForwardResults, theResult.matrix[0]);

        Matrix backwardResult = dense.backward(backwardInput);
        Double[] expectedBackwardResults = new Double[]{-25.5, -33.2};
        assertArrayEquals(expectedBackwardResults, backwardResult.matrix[0]);
        assertArrayEquals(dense.weights.matrix[0], new Double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
        assertArrayEquals(dense.biases.matrix[0], new Double[]{0.0, 0.0, 0.0});

        dense.updateParams();
        assertArrayEquals(Utils.DoubleTodouble(dense.weights.matrix[0]), new double[]{-0.038, -0.64, 5.061, 29.38, 1.19, 4.8}, 0.0001);
        assertArrayEquals(dense.biases.matrix[0], new Double[]{-0.002, 0.069, 0.01});
        

        theResult = dense.forward(forwardInput);
        expectedForwardResults = new Double[]{-271.424, 12688.878, 2098.12};
        assertArrayEquals(expectedForwardResults, theResult.matrix[0]);

        backwardResult = dense.backward(backwardInput);
        expectedBackwardResults = new Double[]{-361.185, -2076.5};
        assertArrayEquals(expectedBackwardResults, backwardResult.matrix[0]);
        assertArrayEquals(Utils.DoubleTodouble(dense.weights.matrix[0]), new double[]{-0.038, -0.64, 5.061, 29.38, 1.19, 4.8}, 0.0001);
        assertArrayEquals(dense.biases.matrix[0], new Double[]{-0.002, 0.069, 0.01});


        dense.updateParams();
        assertArrayEquals(Utils.DoubleTodouble(dense.weights.matrix[0]), new double[]{-0.176, -1.48, 9.822, 58.36, 1.88, 9.0}, 0.0001);
        assertArrayEquals(dense.biases.matrix[0], new Double[]{-0.004, 0.138, 0.02});
    }
}
