package src;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;

public class Layer {
    public Matrix forward(Matrix inputs, int threadIndex) throws Exception {
        return inputs;
    }

    public Matrix backward(Matrix inputs, int threadIndex) {
        return inputs;
    }

    public void updateParams() {
    }

    public void write(int layerIndex, Model model) throws FileNotFoundException, UnsupportedEncodingException {
    }
}