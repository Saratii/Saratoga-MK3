package src;

import java.util.stream.Stream;

public class Softmax{
    Matrix layerOutput;
    public Matrix forward(Matrix inputs){
        Matrix result = new Matrix(inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            double sum = Stream.of(inputs.matrix).map(Math::exp).reduce(0.0, Double::sum);
            result.matrix[i] = Math.exp(inputs.matrix[i]) / sum;
           
        }
        layerOutput = result;
        return result;
    }
    public Matrix backward(Matrix dvalues){
        Matrix result = new Matrix(dvalues.size, dvalues.size);
        for(int i = 0; i < dvalues.size; i++){
            for(int j = 0; j < dvalues.size; j++){
                result.matrix[j*dvalues.size + i] = layerOutput.matrix[i] * ((i == j ? 1 : 0) - layerOutput.matrix[j]);
                
            }
        }
        /*why is it not this?
        Matrix result = new Matrix(dvalues.size, 1);
        for(int i = 0; i < dvalues.size; i++){
            result.matrix[i] = layerOutput.matrix[i] - expectedValues[i];
        }
        */
        
        return result.multiply(dvalues);
    }
}