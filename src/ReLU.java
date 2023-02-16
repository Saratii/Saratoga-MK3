package src;
public class ReLU extends Layer{
    Matrix values;
    public Matrix forward(Matrix values){
        Matrix result = new Matrix(values.z, values.rows, values.cols);
        for(int j = 0; j < values.z; j++){
            for(int i = 0; i < values.rows * values.cols; i++){
                result.matrix[j][i] = (values.matrix[j][i] > 0) ? values.matrix[j][i] : 0.0;
            }
        }
        this.values = result;
        return result;
    }
    public Matrix backward(Matrix dvalues){
        Matrix result = new Matrix(1, dvalues.rows * dvalues.cols, 1);
        for(int i = 0; i < dvalues.size; i++){
            result.matrix[0][i] = (values.matrix[0][i] > 0) ? dvalues.matrix[0][i] : 0.0;
        }
        return result;
    }
}
