import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class Flatten extends Layer {
    int z;
    int rows;
    int cols;
    long[] dims;

    @Override
    public Matrix forward(Matrix inputs, int threadIndex, int batchIndexForThread) {
        INDArray input = inputs.convertToTensor();
        dims = input.shape();
        return Matrix.convertToMatrix(input.reshape(input.length(), 1));
    }

    @Override
    public Matrix backward(Matrix chains, int threadIndex) {
        INDArray chain = chains.convertToTensor();
        return Matrix.convertToMatrix(chain.reshape(dims));
    }

    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.close();
    }
}