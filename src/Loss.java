package src;
public class Loss {
    public static double forward(Matrix yPred, Matrix yTrue) {
        double sum = 0;
        double eps = 1E-15;
        for(int i = 0; i < yPred.size; i++){
            sum += Math.log(yPred.matrix[i] + eps) * yTrue.matrix[i];
        }
        return -sum;
    }
    public static Matrix backward(Matrix yPredicted, Matrix yTrue){
        Matrix results = new Matrix(yTrue.rows, yTrue.cols);
        for(int i = 0; i < yTrue.size; i++){
            results.matrix[i] = -1 * yTrue.matrix[i] / yPredicted.matrix[i];
        }
        return results;
    }
}