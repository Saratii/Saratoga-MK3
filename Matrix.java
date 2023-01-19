

import java.util.Random;

public class Matrix {
    private Random r = new Random();
    private Double[] matrix;
    int rows;
    int cols;
    int size;
    public Matrix(int rows, int cols){
        this.rows = rows;
        this.cols = cols;
        this.size = rows * cols;
        this.matrix = new Double[size];
    }
    @Override
    public String toString() { // returns a string version of the array for viewing
        String s = "[";
        for(int i = 0; i < size - 1; i++){
            s = s + (matrix[i] + ", ");
        }
        return s + matrix[size - 1] + "]";
    }
    public void seed(){ //fills 1d array with random values 0 - 1
        
        for(int i = 0; i < size; i++){
            matrix[i] = r.nextDouble();
        }
    }
    public void testSeed(){ //fills 1d array with random values 0 - 1
        
        for(int i = 0; i < size; i++){
            matrix[i] = (double) i/10;
        }
    }
    public void maxPool(int kernalSize){
        
    }
}
