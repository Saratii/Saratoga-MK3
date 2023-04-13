package tests;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import src.Matrix;
import src.ReLU;

public class reluTest {
    @Test
    public void forward(){
        Matrix a = new Matrix(2, 2, 3);
        a.matrix = new Double[][]{{0.4, -0.1, 0.5, 0.29, -4.0, 0.001}, {-0.2, 0.21, 0.25, 0.129, 4.2, -0.001}};
        ReLU r = new ReLU();
        Matrix b = r.forward(a, 0);
        for(int i = 0; i < b.z; i++){
            for(int j = 0; j < b.rows * b.cols; j++){
                assertEquals(b.matrix[i][j], Double.valueOf(a.matrix[i][j] > 0 ? a.matrix[i][j] : 0.0));
            }
        }
        Matrix c = new Matrix(99, 100, 101);
        c.seed();
        Matrix d = r.forward(c, 0);
        Double min = Double.POSITIVE_INFINITY;
        for(int i = 0; i < c.z; i++){
            for(int j = 0; j < c.rows * c.cols; j++){
                if(d.matrix[i][j] < min){
                    min = d.matrix[i][j];
                }
            }
        }
        assert(0.0 <= min);
        
    }
}
