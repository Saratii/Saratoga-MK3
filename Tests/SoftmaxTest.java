package Tests;

import static org.junit.Assert.assertArrayEquals;

import java.util.stream.Stream;

import org.junit.Test;

import src.Matrix;
import src.Softmax;

public class SoftmaxTest {
    @Test
    public void forward(){
        Matrix inputs = new Matrix(1, 3, 1);
        inputs.matrix = new Double[][]{{1.0, 2.0, -3.0}};
        Matrix result = new Softmax().forward(inputs);
        double[] actualValues = Stream.of(result.matrix[0]).mapToDouble(Double::doubleValue).toArray();
        double[] expectedValues = new double[]{0.267623, 0.72747, 0.0049};
        assertArrayEquals(actualValues, expectedValues, 0.0001);
    }
    @Test
    public void backward(){
        Matrix inputs = new Matrix(1, 3, 1);
        inputs.matrix = new Double[][]{{1.0, 2.0, -3.0}};
        Softmax soft = new Softmax();
        soft.forward(inputs);
        Matrix previousDerivatives = new Matrix(1, 3, 1); 
        previousDerivatives.matrix = new Double[][]{{0.59, 1.85, -69.0}};
        double[] expectedDerivatives = new double[]{-0.15404965, 0.49786693, -0.3438122565};
        Matrix actualDerivativeMatrix = soft.backward(previousDerivatives);
        double[] actualDerivatives = Stream.of(actualDerivativeMatrix.matrix[0]).mapToDouble(Double::doubleValue).toArray();
        assertArrayEquals(actualDerivatives, expectedDerivatives, 0.001);
    
    }
}