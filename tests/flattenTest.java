package tests;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import src.Flatten;
import src.Matrix;

public class flattenTest {
    @Test
    public void forward(){
        Matrix a = new Matrix(6, 7, 8);
        a.seed();
        Flatten f = new Flatten();
        Matrix b = f.forward(a, 0);
        assertEquals(a.size, b.size); 
        Double sum = 0.0;
        for(int i = 0; i < a.z; i++){
            for(int j = 0; j < a.rows * a.cols; j++){
                sum += a.matrix[i][j];
            }
        }
        assertEquals(sum, Double.valueOf(Matrix.sum(b.matrix[0])));

        Matrix c = new Matrix(2, 3, 4);
        c.matrix = new Double[][]{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},{13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}};
        Matrix expected = new Matrix(1, 2 * 3 * 4, 1);
        expected.matrix = new Double[][]{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}};
        for(int i = 0; i < c.size; i++){
            assertEquals(f.forward(c, 0).matrix[0][i], expected.matrix[0][i]);
        }
    }
}
