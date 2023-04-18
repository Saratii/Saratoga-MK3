package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class DenseLayer extends Layer {
    public Matrix weights;
    Matrix[] inputs;
    Matrix[] outputs;
    int NUM_NODES;
    Boolean initialized = false;
    public Matrix biases;
    private Matrix[] biasGradient;
    private Matrix[] weightGradient;

    public DenseLayer(int numInChannels, int NUM_NODES) {
        this.NUM_NODES = NUM_NODES;
        weightGradient = new Matrix[Main.numThreads];
        inputs = new Matrix[Main.numThreads];
        outputs = new Matrix[Main.numThreads];
        for(int i = 0; i < weightGradient.length; i++){
            weightGradient[i] = new Matrix(1, numInChannels, NUM_NODES);
            weightGradient[i].seedZeros();
        }
        biasGradient = new Matrix[Main.numThreads];
        for(int i = 0; i < Main.numThreads; i++){
            biasGradient[i] = new Matrix(1, NUM_NODES, 1);
            biasGradient[i].seedZeros();
            outputs[i] = new Matrix(1, NUM_NODES, 1);
        }
        biases = new Matrix(1, NUM_NODES, 1);
        biases.seedZeros();
        weights = new Matrix(1, numInChannels, NUM_NODES);
        weights.seedUniform();
    }

    @Override
    public Matrix forward(Matrix inputs, int threadIndex) throws Exception {
        if(inputs.rows != weights.rows){
            throw new Exception("invalid input channels in dense layer, expected " + inputs.rows);
        }
        if(inputs.z != 1){
            throw new Exception("flatten your input ");
        }
        this.inputs[threadIndex] = inputs;
        for(int i = 0; i < NUM_NODES; i++){
            double sum = biases.matrix[0][i];
            for(int j = 0; j < inputs.innerSize; j++){
                sum += inputs.matrix[0][j] * weights.matrix[0][i * inputs.innerSize + j];
            }
            outputs[threadIndex].matrix[0][i] = sum;
        }
        return outputs[threadIndex];
    }

    @Override
    public Matrix backward(Matrix previousDerivatives, int threadIndex) {
        biasGradient[threadIndex].add(previousDerivatives);
        Matrix passedOnDerivatives = new Matrix(1, inputs[threadIndex].size, 1);
        for(int i = 0; i < inputs[threadIndex].size; i++){
            passedOnDerivatives.matrix[0][i] = 0.0;
            for(int j = 0; j < NUM_NODES; j++){
                passedOnDerivatives.matrix[0][i] += previousDerivatives.matrix[0][j] * weights.matrix[0][j * inputs[threadIndex].size + i];
                weightGradient[threadIndex].matrix[0][j * inputs[threadIndex].size + i] += previousDerivatives.matrix[0][j] * inputs[threadIndex].matrix[0][i];
            }
        }
        return passedOnDerivatives;
    }

    @Override
    public void updateParams() {
        for(int j = 0; j < biasGradient.length; j++){
            for(int i = 0; i < biasGradient[j].size; i++){
                biases.matrix[0][i] -= biasGradient[j].matrix[0][i] * Main.ALPHA;
            }
            biasGradient[j].seedZeros();
        }
        for(int j = 0; j < weightGradient.length; j++){
            for(int i = 0; i < weightGradient[j].size; i++){
                weights.matrix[0][i] -= weightGradient[j].matrix[0][i] * Main.ALPHA;
            }
            weightGradient[j].seedZeros();
        }
    }

    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.println("Total Parameters{" + (weights.size + biases.size) + "}");
        writer.println("Number of Nodes{" + NUM_NODES + "}\n");
        writer.println("Number of Inputs{" + (inputs[0].size) + "}");
        writer.println(biases.toString(false) + "\n");
        writer.println("Number of weights{" + inputs[0].size + ", " + NUM_NODES + "}\n");
        writer.println(weights.toString(false));
        writer.close();
    }
}
