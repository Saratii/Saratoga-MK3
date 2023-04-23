import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

public class Model {
    public ArrayList<Layer> layers = new ArrayList<>();
    Matrix[] inputs;
    Matrix[] expected;
    boolean isClassifying;

    public Model() {
        expected = new Matrix[Main.numThreads];
        inputs = new Matrix[Main.numThreads];
    }

    public double forward(INDArray input, Matrix expected, int threadIndex, int batchIndexForThread) throws Exception {
        for(int i = 0; i < layers.size(); i++){
            layers.get(i).isClassifying = isClassifying;
            input = layers.get(i).forward(input, threadIndex, batchIndexForThread);
        }
        double l = Loss.forward(input, expected);
        this.expected[threadIndex] = expected;
        this.inputs[threadIndex] = Matrix.convertToMatrix(input);
        return l;
    }

    public void backward(int threadIndex) {
        Matrix gradients = Loss.backward(inputs[threadIndex], expected[threadIndex]);
        for(int i = 0; i < layers.size(); i++){
            gradients = layers.get(layers.size() - 1 - i).backward(gradients, threadIndex);
        }
    }

    public void updateParams() {
        for(Layer layer : layers){
            layer.updateParams();
        }
    }

    public void write() throws IOException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-architecture", "UTF-8");
        for(int i = 0; i < layers.size(); i++){
            layers.get(i).write();
            writer.println(layers.get(i));
        }
        writer.close();
    }
}