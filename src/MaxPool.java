package src;

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
        int index = 0;
        for(int l = 0; l < input.z; l++){
            for(int i = 0; i < input.rows; i += kernalSize){
                for(int j = 0; j < input.cols; j += kernalSize){
                    double maxVal = input.matrix[l][input.convert(i, j)];
                    for(int k = 0; k < kernalSize && i + k < input.rows; k++){
                        for(int h = 0; h < kernalSize && j + h < input.cols; h++){
                            if(input.matrix[l][input.convert(i + k, j + h)] > maxVal){
                                maxVal = input.matrix[l][input.convert(i + k, j + h)];
                                gradients.matrix[l][input.convert(i + k, j + h)] = 1.0;
                            } else {
                                gradients.matrix[l][input.convert(i + k, j + h)] = 0.0;
                            }
                        } 
                    }
                    result.matrix[index][result.convert(i / kernalSize, j / kernalSize)] = maxVal;
                }
            }
            index++;
        }
        return result;
    }
    public Matrix backward(Matrix prevGradients){
        int i = 0;
        for(int j = 0; j < gradients.z; j++){
            for(int k = 0; k < gradients.rows * gradients.cols; k++){
                gradients.matrix[j][k] *= prevGradients.matrix[j][i];
                i++;
                if(i == kernalSize){
                    i = 0;
                }
            }
        }
        return gradients;
    }
}


