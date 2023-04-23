import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

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

    public static INDArray backward(INDArray yPredicted, Matrix yTrue) {
        INDArray result = Nd4j.create(DataType.DOUBLE, yTrue.rows, yTrue.cols);
        for(int i = 0; i < yTrue.size; i++){
            result.putScalar(i, -yTrue.matrix[0][i] / (yPredicted.getDouble(i) + eps));
        }
        return result;
    }
}
