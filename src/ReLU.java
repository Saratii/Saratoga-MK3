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
    public Matrix backward(Matrix dvalues){
        Matrix result = new Matrix(dvalues.size, 1);
        for(int i = 0; i < dvalues.size; i++){
            result.matrix[i] = (values.matrix[i] > 0) ? dvalues.matrix[i] : 0.0;
        }
        return result;
    }
}
