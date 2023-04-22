public class Loss extends Layer {
    static double eps = 1E-15;

    public static double forward(Matrix yPred, Matrix yTrue) throws Exception {
        if(yPred.z != yTrue.z || yPred.innerSize != yTrue.innerSize){
            throw new Exception("Invalid resultant tensor size: " + yPred.getSize());
        }
        double sum = 0;
        for(int i = 0; i < yPred.size; i++){
            sum += Math.log(yPred.matrix[0][i] + eps) * yTrue.matrix[0][i];
        }
        return -sum;
    }

    public static Matrix backward(Matrix yPredicted, Matrix yTrue) {
        Matrix results = new Matrix(1, yTrue.rows, yTrue.cols);
        for(int i = 0; i < yTrue.size; i++){
            results.matrix[0][i] = -1 * yTrue.matrix[0][i] / (yPredicted.matrix[0][i] + eps);
        }
        return results;
    }
}
