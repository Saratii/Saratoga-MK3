package src;
public class ReLU {
    Matrix layerOutput;
    public Matrix forward(Matrix values){
        Matrix result = new Matrix(values.rows, values.cols);
        for(int i = 0; i < values.size; i++){
            result.matrix[i] = (values.matrix[i] > 0) ? values.matrix[i] : 0.0;
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
        return result;
    }
}
