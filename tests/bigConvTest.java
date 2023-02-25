package tests;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

import src.Matrix;

public class bigConvTest {
    @Test
    public void test(){
        Matrix a = new Matrix(1, 2, 2);
        Matrix b = new Matrix(1, 2, 2);
        a.matrix = new Double[][]{{1.0, 2.0, 3.0, 4.0}};
        b.matrix = new Double[][]{{5.0, 6.0, 7.0, 8.0}};
        Matrix c = a.bigConvolution(b);
        assertArrayEquals(new Double[]{8.0, 23.0, 14.0, 30.0, 70.0, 38.0, 18.0, 39.0, 20.0}, c.matrix[0]);
    }
}
