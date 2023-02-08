package src;
public class ReLU {
    Matrix values;
    public Matrix forward(Matrix values){
        Matrix result = new Matrix(values.rows, values.cols);
        for(int i = 0; i < values.size; i++){
            result.matrix[i] = (values.matrix[i] > 0) ? values.matrix[i] : 0.0;
        }
        this.values = result;
        return result;
    }
    public Matrix[] forward(Matrix[] values){
        Matrix[] result = new Matrix[values.length];
        for(int i = 0; i < values.length; i++){
            Matrix featureMap = new Matrix(values[0].rows, values[0].cols);
            for(int j = 0; j < values[i].size; j++){
                featureMap.matrix[j] = (values[i].matrix[j] > 0) ? values[i].matrix[j] : 0.0;
            }
            result[i] = featureMap;
        }
        return result;
    }
    public Matrix backward(Matrix dvalues){
        Matrix result = new Matrix(dvalues.size, 1);
        for(int i = 0; i < dvalues.size; i++){
            result.matrix[i] = (values.matrix[i] > 0) ? dvalues.matrix[i] : 0.0;
        }
        return result;
    }
}
