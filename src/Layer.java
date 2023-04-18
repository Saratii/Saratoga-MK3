package src;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;

public class Layer {
    boolean isClassifying = false;
    public Matrix forward(Matrix inputs, int threadIndex) throws Exception {
        return inputs;
    }

    public Matrix backward(Matrix inputs, int threadIndex) {
        return inputs;
    }

    public void updateParams() {
    }

    public void write() throws FileNotFoundException, UnsupportedEncodingException {
    }
}