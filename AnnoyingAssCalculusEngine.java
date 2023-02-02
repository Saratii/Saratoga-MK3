public class AnnoyingAssCalculusEngine {
    public static double loss(Matrix yPred, Matrix yTrue) {
        double sum = 0;
        for(int i = 0; i < yPred.size; i++){
            sum += Math.log(yPred.matrix[i]) * yTrue.matrix[i];
        }
        return -sum;
    }
    //weights = weights - learningRate * dW
    //dW = dL/dz * dz/dw
    //dL/dz = yPred - yTrue
    public static Matrix dldz(Matrix yPred, Matrix yTrue){
        Matrix result = new Matrix(yPred.size, 1);
        for(int i = 0; i < yPred.size; i++){
            result.matrix[i] = yPred.matrix[i] - yTrue.matrix[i];
        }
        return result;
    }
    public static Matrix dzdw(Matrix yPred, Matrix yTrue) {
        Matrix dSdW = new Matrix(yPred.size * yPred.size, 1);
        for (int i = 0; i < yPred.size; i++) {
            for (int j = 0; j < yPred.size; j++) {
                dSdW.matrix[i * yPred.size + j] = yPred.matrix[i] * (i == j ? 1 - yPred.matrix[j] : -yPred.matrix[j]);
            }
        }
        return dSdW;
    }
    public static Matrix updateWeights(Matrix weights, double alpha, Matrix dldz, Matrix dzdw){
        Matrix result = new Matrix(weights.rows, weights.cols);
        for(int i = 0; i < result.size; i++){
            result.matrix[i] = weights.matrix[i] - alpha * dldz.matrix[i] * dzdw.matrix[i];
        }
        return result;
    }

}
