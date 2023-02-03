package src;
public class Loss {
    public static double calcLoss(Matrix yPred, Matrix yTrue) {
        double sum = 0;
        for(int i = 0; i < yPred.size; i++){
            sum += Math.log(yPred.matrix[i]) * yTrue.matrix[i];
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