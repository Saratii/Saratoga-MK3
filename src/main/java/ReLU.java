import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.function.DoubleUnaryOperator;

public class ReLU extends Layer {
    Matrix[] values;

    public ReLU() {
        values = new Matrix[Main.numThreads];
    }

    @Override
    public Matrix forward(Matrix value, int threadIndex, int batchIndexForThread) {
        INDArray input = value.convertToTensor();
        for(int i = 0; i < input.length(); i++){
            input.putScalar(i, Math.max(input.getDouble(i), 0));
        }
        values[threadIndex] = Matrix.convertToMatrix(input);
        return values[threadIndex];
    }

    @Override
    public Matrix backward(Matrix dvalues, int threadIndex) {
        INDArray chain = dvalues.convertToTensor();
        INDArray inputs = values[threadIndex].convertToTensor();
        for(int i = 0; i < chain.length(); i++){
            chain.putScalar(i, inputs.getDouble(i) > 0 ? chain.getDouble(i) : 0.0);
        }
        return Matrix.convertToMatrix(chain);
    }

    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.close();
    }
}