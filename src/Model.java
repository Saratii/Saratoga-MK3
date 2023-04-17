package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

public class Model {
    public ArrayList<Layer> layers = new ArrayList<>();
    Matrix[] inputs;
    Matrix[] expected;
    Boolean profiling = false;

    public Model() {
        expected = new Matrix[Main.numThreads];
        inputs = new Matrix[Main.numThreads];
    }

    public double forward(Matrix inputs, Matrix expected, int threadIndex) throws Exception {
        for(int i = 0; i < layers.size(); i++){
            inputs = layers.get(i).forward(inputs, threadIndex);
        }
        double l = Loss.forward(inputs, expected);
        this.expected[threadIndex] = expected;
        this.inputs[threadIndex] = inputs;
        return l;
    }

    public void backward(int threadIndex) {
        Matrix gradients;
        gradients = Loss.backward(inputs[threadIndex], expected[threadIndex]);
        for(int i = 0; i < layers.size(); i++){
            gradients = layers.get(layers.size() - 1 - i).backward(gradients, threadIndex);
        }
    }

    public void updateParams() {
        for(Layer layer : layers){
            layer.updateParams();
        }
    }

    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("logs/log-architecture", "UTF-8");
        for(int i = 0; i < layers.size(); i++){
            layers.get(i).write(i, this);
            writer.println(layers.get(i));
        }
        writer.close();
    }
}