package src;

public class Softmax extends Layer{
    Matrix layerOutput;
    public Matrix forward(Matrix inputs){
        Matrix result = new Matrix(1, inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            double shiftConstant = Matrix.maxValue(inputs.matrix[0]);
            double sum = 0.0;
            for(int j = 0; j < inputs.size; j++){
                sum+= Math.exp(inputs.matrix[0][j] - shiftConstant);
            }
           
            result.matrix[0][i] = Math.exp(inputs.matrix[0][i] - shiftConstant) / sum;
           
        }
        layerOutput = result;
        return result;
    }
    public Matrix backward(Matrix dvalues){
        Matrix result = new Matrix(1, dvalues.size, dvalues.size);
        for(int i = 0; i < dvalues.size; i++){
            for(int j = 0; j < dvalues.size; j++){
                result.matrix[0][j*dvalues.size + i] = layerOutput.matrix[0][i] * ((i == j ? 1 : 0) - layerOutput.matrix[0][j]);
                
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