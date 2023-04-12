package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class MaxPool extends Layer{
    int kernalSize;
    int z;
    int rows;
    int cols;
    Matrix input;
    Matrix gradients;
    public MaxPool(int kernalSize){
        this.kernalSize = kernalSize;
    }
    public Matrix forward(Matrix input){
        this.z = input.z;
        this.rows = input.rows;
        this.cols = input.cols;
        this.input = input;
        gradients = new Matrix(input.z, input.rows, input.cols);
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
                            gradients.matrix[l][input.convert(i + k, j + h)] = 0.0;
                        } 
                    }
                    result.matrix[l][result.convert(i / kernalSize, j / kernalSize)] = maxVal;
                    gradients.matrix[l][input.convert(i + maxK, j + maxH)] = 1.0;
                }
            }
        }
        return result;
    }
    public Matrix backward(Matrix prevGradients){
        int i = 0;
        for(int j = 0; j < gradients.z; j++){//why only iterate up to gradients.z when higher values are accepted
            for(int k = 0; k < gradients.rows * gradients.cols; k++){
                gradients.matrix[j][k] *= prevGradients.matrix[j][prevGradients.convert(k / gradients.cols / kernalSize, (k%gradients.cols) / kernalSize)];
                i++;
                if(i == kernalSize){
                    i = 0;
                }
            }
        }
        return gradients;
    }
    public void write(int layerIndex, Model model) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("logs/log-" + model.layers.get(layerIndex), "UTF-8");
        writer.println(model.layers.get(layerIndex));
        writer.println("Size of Kernals{" + kernalSize + "}");
        writer.close();
    }
}


