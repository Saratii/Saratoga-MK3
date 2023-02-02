public class Softmax{
    public static Matrix forward(Matrix inputs){
        Matrix results = new Matrix(inputs.size, 1);
        double sum = 0;
        for( int i = 0; i < inputs.size; i++){
            results.matrix[i] = Math.exp(inputs.matrix[i]);
            sum += results.matrix[i];
        }
        for(int i = 0; i < results.size; i++){
            results.matrix[i] /= sum;
        }
        return results;
    }
    public static Matrix backward(Matrix dvalues){
        Matrix result = new Matrix(dvalues.rows, dvalues.cols);
        return result;
    }
}