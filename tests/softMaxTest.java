package tests;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import src.Matrix;
import src.Softmax;

public class softMaxTest {
    @Test
    public void forward(){
        Matrix a = new Matrix(1, 3, 3);
        a.matrix = new Double[][]{{1.0, 0.99, 2.9, -1.92, 1.1, -0.0001, -0.2, -1.3, 6.0}};
        Matrix expected = new Matrix(1, 3, 3);
        Softmax s = new Softmax();
        Matrix b = s.forward(a, 0);
        expected.matrix = new Double[][]{{0.00629, 0.00623, 0.042, 0.00034, 0.00695, 0.00231, 0.00189, 0.00063, 0.93331}};
        for(int i = 0; i < a.size; i++){
            assertEquals(expected.matrix[0][i], b.matrix[0][i], 0.0001);
        }
        assertEquals(Double.valueOf(1.0), Double.valueOf(Matrix.sum(b.matrix[0])), 0.000001);
        Matrix c = new Matrix(1, 68, 69);
        c.seed();
        Matrix d = s.forward(c, 0);
        Double sum = 0.0;
        for(int i = 0; i < c.size; i++){
            sum += Math.exp(c.matrix[0][i]);
        }
        for(int i = 0; i < c.size; i++){
            assertEquals(d.matrix[0][i], Double.valueOf(Math.exp(c.matrix[0][i]) / sum), 0.000001);
        }
        assertEquals(Double.valueOf(Matrix.sum(d.matrix[0])), Double.valueOf(1.0), 0.000001);
    }
}
