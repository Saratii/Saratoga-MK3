public class ReLU {
    public static Matrix forward(Matrix values){
        Matrix result = new Matrix(values.rows, values.cols);
        for(int i = 0; i < values.size; i++){
            result.matrix[i] = (values.matrix[i] > 0) ? values.matrix[i] : 0.0;
        }
    return result;
    }
    public static Matrix backward(Matrix dvalues){
        Matrix result = new Matrix(dvalues.rows, dvalues.cols);
        for(int i = 0; i < dvalues.size; i++){
            result.matrix[i] = (dvalues.matrix[i] > 0) ? 1.0: 0.0;
        }
        return result;
    }
}
