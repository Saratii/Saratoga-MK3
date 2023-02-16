package src;
public class Loss extends Layer{
    public static double forward(Matrix yPred, Matrix yTrue) {
        double sum = 0;
        double eps = 1E-15;
        for(int i = 0; i < yPred.size; i++){
            sum += Math.log(yPred.matrix[0][i] + eps) * yTrue.matrix[0][i];
        }
        return -sum;
    }
    public static Matrix backward(Matrix yPredicted, Matrix yTrue){
        Matrix results = new Matrix(1, yTrue.rows, yTrue.cols);
        for(int i = 0; i < yTrue.size; i++){
            results.matrix[0][i] = -1 * yTrue.matrix[0][i] / yPredicted.matrix[0][i];
        }
        return results;
    }
}