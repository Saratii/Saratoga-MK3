package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class DenseLayer extends Layer{
    public Matrix weights;
    Matrix[] inputs;
    int NUM_NODES;
    Boolean initialized = false;
    public Matrix biases;
    Matrix[] weightGradientsPerThread;
    Matrix[] biasGradientsPerThread;
    public DenseLayer(int numInChannels, int numOutChannels) {
        this.NUM_NODES = numOutChannels;
        biases = new Matrix(1, NUM_NODES, 1);
        biases.seedZeros();
        weights = new Matrix(1, numInChannels, NUM_NODES);
        weights.seedUniform();
        inputs = new Matrix[Main.numThreads];
        weightGradientsPerThread = new Matrix[Main.numThreads];
        biasGradientsPerThread = new Matrix[Main.numThreads];
    }
    @Override
    public Matrix forward(Matrix inputs, int threadIndex){
        Matrix outputs = new Matrix(1, NUM_NODES, 1);
        this.inputs[threadIndex] = inputs;
        for(int i = 0; i < NUM_NODES; i++){
            double sum = biases.matrix[0][i];
            for(int j = 0; j < inputs.rows * inputs.cols; j++){
                sum += inputs.matrix[0][j] * weights.matrix[0][i * inputs.rows * inputs.cols + j];
            }
            outputs.matrix[0][i] = sum;
        }
        return outputs;
    }
    @Override
    public Matrix backward(Matrix previousDerivatives, int threadIndex){
        biasGradientsPerThread[threadIndex] = previousDerivatives;
        Matrix weightGradient = new Matrix(1, inputs[threadIndex].size, NUM_NODES);
        Matrix passedOnDerivatives = new Matrix(1, inputs[threadIndex].size, 1);
        for(int i = 0; i < inputs[threadIndex].size; i++){
            passedOnDerivatives.matrix[0][i] = 0.0;
            for(int j = 0; j < NUM_NODES; j++){
                passedOnDerivatives.matrix[0][i] += previousDerivatives.matrix[0][j] * weights.matrix[0][j * inputs[threadIndex].size + i];
                weightGradient.matrix[0][j * inputs[threadIndex].size + i] = previousDerivatives.matrix[0][j] * inputs[threadIndex].matrix[0][i];
            }
        }
        weightGradientsPerThread[threadIndex] = weightGradient;
        return passedOnDerivatives;
    }
    @Override
    public void updateParams(){
        for(int j = 0; j < biasGradientsPerThread.length; j++){
            for(int i = 0; i < biasGradientsPerThread[j].size; i++){
                biases.matrix[0][i] -= biasGradientsPerThread[j].matrix[0][i] * Main.ALPHA;
            }
        }
        for(int j = 0; j < weightGradientsPerThread.length; j++){
            for(int i = 0; i < weightGradientsPerThread[j].size; i++){
                weights.matrix[0][i] -= weightGradientsPerThread[j].matrix[0][i] * Main.ALPHA;
            }
        }
    }
    @Override
    public void write(int layerIndex, Model model) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("logs/log-" +  model.layers.get(layerIndex), "UTF-8");
        writer.println(model.layers.get(layerIndex));
        writer.println("Total Parameters{" + (weights.size + biases.size) + "}");
        writer.println("Number of Nodes{" + NUM_NODES + "}\n");
        writer.println("Number of Inputs{" + (inputs[0].size) + "}");
        writer.println(biases.toString(false) + "\n");
        writer.println("Number of weights{" + inputs[0].size + ", " + NUM_NODES + "}\n");
        writer.println(weights.toString(false));
        writer.close();
    }
}