package Tests;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import java.util.stream.Stream;
import org.junit.Test;
import src.ConvolutionLayer;
import src.Matrix;

public class PytorchTest {
    @Test
    public void forward(){
        Matrix input = new Matrix(4, 4);
        ConvolutionLayer conv = new ConvolutionLayer(3, 1, 2);
        input.matrix = new Double[]{-0.0885,  0.5239, -0.6659,  0.8504, -1.3527, -1.6959,  0.5667,  0.7935, -0.1932, -0.3090,  0.5026, -0.8594, 0.7502, -0.5855, -0.1734, 0.1835};
        System.out.println(input.sum());
        assertEquals(-1.7528, input.sum(), 0.0001); //same starting number
        conv.kernals[0].matrix = new Double[]{-0.0037,  0.2682, -0.4115, -0.3680};
        conv.kernals[1].matrix = new Double[]{-0.1926,  0.1341, -0.0099,  0.3964};
        conv.kernals[2].matrix = new Double[]{-0.0444,  0.1323, -0.1511, -0.0983};
        Matrix[] convOut = conv.forward(input);
        double[] expectedSet1 = new double[]{1.3215,  0.3088, -0.2946, -0.2566,  0.1006,  0.3201, -0.1754,  0.4407, -0.2286};
        double[] expectedSet2 = new double[]{-0.5717,  0.0513,  0.5512, -0.0875,  0.6049, -0.3484, -0.2438,  0.0639, -0.1376};
        double[] expectedSet3 = new double[]{ 0.4443,  0.0892, -0.0216, -0.1048,  0.1475,  0.0884, -0.0881,  0.1857, -0.1278};
        double[] actualSet1 = Stream.of(convOut[0].matrix).mapToDouble(Double::doubleValue).toArray();
        double[] actualSet2 = Stream.of(convOut[1].matrix).mapToDouble(Double::doubleValue).toArray();
        double[] actualSet3 = Stream.of(convOut[2].matrix).mapToDouble(Double::doubleValue).toArray();

        assertArrayEquals(expectedSet1, actualSet1, 0.001);
        assertArrayEquals(expectedSet2, actualSet2, 0.001);
        assertArrayEquals(expectedSet3, actualSet3, 0.001); //verifying convolution works


        // -0.0037 * 0.8504 + 0.2682 * -1.3527 + -0.4115 * 0.7935 + -0.3680 * 0.5988

        // -0.0885 + 0.5239 -0.6659+  0.8504+ -1.3527+ -1.6959+  0.5667+  0.7935+ -0.1932+ -0.3090+  0.5026+ -0.8594+ 0.7502+ -0.5855+ -0.1734+ 0.1835
    }
}