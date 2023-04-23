import org.nd4j.linalg.api.ndarray.INDArray;

public class Loss {
    static double eps = 1E-15;

    public static double forward(INDArray yPred, Matrix yTrue) throws Exception {
        if(yPred.shape().length != 2 || yPred.size(0) != yTrue.innerSize){
            throw new Exception("Invalid resultant tensor size: " + yPred.shape());
        }
        double sum = 0;
        for(int i = 0; i < yPred.length(); i++){
            sum += Math.log(yPred.getDouble(i) + eps) * yTrue.matrix[0][i];
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
