import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

public class Layer {
    boolean isClassifying = false;
    public INDArray forward(INDArray inputs, int threadIndex, int batchIndexForThread) throws Exception {
        return inputs;
    }

    public Matrix backward(Matrix inputs, int threadIndex) {
        return inputs;
    }

    public void updateParams() {
    }

    public void write() throws IOException {
    }
}