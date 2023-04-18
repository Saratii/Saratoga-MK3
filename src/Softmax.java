package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class Softmax extends Layer {
    Matrix[] layerOutput;

    public Softmax() {
        layerOutput = new Matrix[Main.numThreads];
    }

    @Override
    public Matrix forward(Matrix inputs, int threadIndex) {
        Matrix result;
        result = new Matrix(1, inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            double shiftConstant = MathUtils.maxValue(inputs.matrix[0]);
            double sum = 0.0;
            for(int j = 0; j < inputs.size; j++){
                sum += Math.exp(inputs.matrix[0][j] - shiftConstant);
            }
            result.matrix[0][i] = Math.exp(inputs.matrix[0][i] - shiftConstant) / (sum);
        }
        layerOutput[threadIndex] = result;
        return result;
    }

    @Override
    public Matrix backward(Matrix dvalues, int threadIndex) {
        Matrix result = new Matrix(1, dvalues.size, dvalues.size);
        for(int i = 0; i < dvalues.size; i++){
            for(int j = 0; j < dvalues.size; j++){
                result.matrix[0][j * dvalues.size + i] = layerOutput[threadIndex].matrix[0][i] * ((i == j ? 1 : 0) - layerOutput[threadIndex].matrix[0][j]);
            }
        }
        return result.multiply(dvalues);
    }
    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.close();
    }
}