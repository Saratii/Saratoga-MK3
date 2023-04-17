package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class ReLU extends Layer {
    Matrix[] values;

    public ReLU() {
        values = new Matrix[Main.numThreads];
    }

    @Override
    public Matrix forward(Matrix values, int threadIndex) {
        Matrix result = new Matrix(values.z, values.rows, values.cols);
        for(int j = 0; j < values.z; j++){
            for(int i = 0; i < values.rows * values.cols; i++){
                result.matrix[j][i] = (values.matrix[j][i] > 0) ? values.matrix[j][i] : 0.0;
            }
        }
        this.values[threadIndex] = result;
        return result;
    }

    @Override
    public Matrix backward(Matrix dvalues, int threadIndex) {
        Matrix result = new Matrix(dvalues.z, dvalues.rows, dvalues.cols);
        for(int j = 0; j < dvalues.z; j++){
            for(int i = 0; i < dvalues.rows * dvalues.cols; i++){
                result.matrix[j][i] = (values[threadIndex].matrix[j][i] > 0) ? dvalues.matrix[j][i] : 0.0;
            }
        }
        return result;
    }

    @Override
    public void write(int layerIndex, Model model) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("logs/log-" + model.layers.get(layerIndex), "UTF-8");
        writer.println(model.layers.get(layerIndex));
        writer.close();
    }
}