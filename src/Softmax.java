package src;

import java.util.stream.Stream;

public class Softmax{
    Matrix layerOutput;
    public Matrix forward(Matrix inputs){
        Matrix result = new Matrix(inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            result.matrix[i] = Math.exp(inputs.matrix[i]) / Stream.of(inputs.matrix).map(Math::exp).reduce(0.0, Double::sum);
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
        
        return result.multiply(dvalues);
    }
}