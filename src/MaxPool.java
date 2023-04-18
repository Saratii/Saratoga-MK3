package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class MaxPool extends Layer {
    final int kernalSize;
    Matrix[] gradients;

    public MaxPool(int kernalSize) {
        this.kernalSize = kernalSize;
        gradients = new Matrix[Main.numThreads];
    }

    @Override
    public Matrix forward(Matrix input, int threadIndex) {
        gradients[threadIndex] = new Matrix(input.z, input.rows, input.cols);
        int resultSize = (int) Math.ceil((float) input.rows / (float) kernalSize);
        Matrix result = new Matrix(input.z, resultSize, resultSize);
        for(int l = 0; l < input.z; l++){
            for(int i = 0; i < input.rows; i += kernalSize){
                for(int j = 0; j < input.cols; j += kernalSize){
                    double maxVal = input.matrix[l][input.convert(i, j)];
                    int maxK = 0;
                    int maxH = 0;
                    for(int k = 0; k < kernalSize && i + k < input.rows; k++){
                        for(int h = 0; h < kernalSize && j + h < input.cols; h++){
                            if(input.matrix[l][input.convert(i + k, j + h)] > maxVal){
                                maxVal = input.matrix[l][input.convert(i + k, j + h)];
                                maxK = k;
                                maxH = h;
                            }
                            gradients[threadIndex].matrix[l][input.convert(i + k, j + h)] = 0.0;
                        }
                    }
                    result.matrix[l][result.convert(i / kernalSize, j / kernalSize)] = maxVal;
                    gradients[threadIndex].matrix[l][input.convert(i + maxK, j + maxH)] = 1.0;
                }
            }
        }
        return result;
    }

    @Override
    public Matrix backward(Matrix prevGradients, int threadIndex) {
        int i = 0;
        for(int j = 0; j < gradients[threadIndex].z; j++){
            for(int k = 0; k < gradients[threadIndex].innerSize; k++){
                gradients[threadIndex].matrix[j][k] *= prevGradients.matrix[j][prevGradients.convert(k / gradients[threadIndex].cols / kernalSize,
                        (k % gradients[threadIndex].cols) / kernalSize)];
                i++;
                if(i == kernalSize){
                    i = 0;
                }
            }
        }
        return gradients[threadIndex];
    }

    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.println("Size of Kernals{" + kernalSize + "}");
        writer.close();
    }
}