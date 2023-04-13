package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class Flatten extends Layer{
    int z;
    int rows;
    int cols;
    @Override
    public Matrix forward(Matrix input, int threadIndex){
        Matrix result = new Matrix(1, input.z * input.cols * input.rows, 1);
        this.z = input.z;
        this.rows = input.rows;
        this.cols = input.cols;
        for(int i = 0; i < input.z; i++){
            for(int j = 0; j < input.rows * input.cols; j++){
                result.matrix[0][i * input.rows * input.cols + j] = input.matrix[i][j];
            }
        }
        return result;
    }
    @Override
    public Matrix backward(Matrix notInput, int threadIndex){
        Matrix result = new Matrix(z, rows, cols);
        for(int i = 0; i < z; i++){
            for(int j = 0; j < rows * cols; j++){
                result.matrix[i][j] = notInput.matrix[0][i * rows * cols + j];
            }
        }
        return result;
    }
    @Override
    public void write(int layerIndex, Model model) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("logs/log-" +  model.layers.get(layerIndex), "UTF-8");
        writer.println(model.layers.get(layerIndex));
        writer.close();
    }
}