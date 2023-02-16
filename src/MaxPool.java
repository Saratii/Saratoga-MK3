package src;

public class MaxPool extends Layer{
    int kernalSize;
    public MaxPool(int kernalSize){
        this.kernalSize = kernalSize;
    }
    public Matrix forward(Matrix input){
        int resultSize = (int) Math.ceil((float) input.rows / (float) kernalSize);
        Matrix result = new Matrix(input.z, resultSize, resultSize);
        int index = 0;
        for(Double[] d: input.matrix){
            for(int i = 0; i < input.rows; i += kernalSize){
                for(int j = 0; j < input.cols; j += kernalSize){
                    double maxVal = d[input.convert(i, j)];
                    for(int k = 0; k < kernalSize && i + k < input.rows; k++){
                        for(int h = 0; h < kernalSize && j + h < input.cols; h++){
                            if(d[input.convert(i + k, j + h)] > maxVal){
                                maxVal = d[input.convert(i + k, j + h)];
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
}

