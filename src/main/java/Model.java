import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class Model {
    public ArrayList<Layer> layers = new ArrayList<>();
    INDArray[] inputs = new INDArray[Main.numThreads];
    Matrix[] expected = new Matrix[Main.numThreads];
    boolean isClassifying;

    public double forward(INDArray input, Matrix expected, int threadIndex, int batchIndexForThread) throws Exception {
        for(int i = 0; i < layers.size(); i++){
            layers.get(i).isClassifying = isClassifying;
            input = layers.get(i).forward(input, threadIndex, batchIndexForThread);
        }
        double l = Loss.forward(input, expected);
        this.expected[threadIndex] = expected;
        this.inputs[threadIndex] = input;
        return l;
    }

    public void backward(int threadIndex) {
        INDArray gradient = Loss.backward(inputs[threadIndex], expected[threadIndex]);
        for(int i = 0; i < layers.size(); i++){
            gradient = layers.get(layers.size() - 1 - i).backward(gradient, threadIndex);
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