package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;

public class DenseLayer extends Layer{
    public Matrix weights;
    Matrix inputs;
    Matrix outputs;
    int NUM_NODES;
    Boolean initialized = false;
    public Matrix biases;
    private Matrix[] biasGradient;
    private Matrix[] weightGradient;
    public DenseLayer(int numInChannels, int NUM_NODES) {
        this.NUM_NODES = NUM_NODES;
        weightGradient = new Matrix[Main.numThreads];
        for(int i = 0; i < weightGradient.length; i++){
            weightGradient[i] = new Matrix(1, numInChannels, NUM_NODES);
            weightGradient[i].seedZeros();
        }
        biasGradient = new Matrix[Main.numThreads];
        for(int i = 0; i < Main.numThreads; i++){
            biasGradient[i] = new Matrix(1, NUM_NODES, 1);
            biasGradient[i].seedZeros();
        }
        biases = new Matrix(1, NUM_NODES, 1);
        biases.seedZeros();
        weights = new Matrix(1, numInChannels, NUM_NODES);
        weights.seedUniform();
        outputs = new Matrix(1, NUM_NODES, 1);
    }
    @Override
    public Matrix forward(Matrix inputs, int threadIndex) throws Exception{
        if(inputs.rows != weights.rows){
            throw new Exception("invalid input channels in dense layer, expected " + inputs.rows);
        }
        if(inputs.z != 1){
            throw new Exception("flatten your input ");
        }
        this.inputs = inputs;
        for(int i = 0; i < NUM_NODES; i++){
            double sum = biases.matrix[0][i];
            for(int j = 0; j < inputs.rows * inputs.cols; j++){
                sum += inputs.matrix[0][j] * weights.matrix[0][i * inputs.rows * inputs.cols + j];
            }
            outputs.matrix[0][i] = sum;
            if(Double.isNaN(outputs.matrix[0][i])){
                Arrays.toString(inputs.matrix[0]);
                Arrays.toString(weights.matrix[0]);
            }
        }
        return outputs; 
    }
    @Override
    public Matrix backward(Matrix previousDerivatives, int threadIndex){
        biasGradient[threadIndex].add(previousDerivatives);
        Matrix passedOnDerivatives = new Matrix(1, inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            passedOnDerivatives.matrix[0][i] = 0.0;
            for(int j = 0; j < NUM_NODES; j++){
                passedOnDerivatives.matrix[0][i] += previousDerivatives.matrix[0][j] * weights.matrix[0][j * inputs.size + i];
                weightGradient[threadIndex].matrix[0][j * inputs.size + i] += previousDerivatives.matrix[0][j] * inputs.matrix[0][i];
            }
        }
        return passedOnDerivatives;
    }
    @Override
    public void updateParams(){
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
    public void write(int layerIndex, Model model) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("logs/log-" +  model.layers.get(layerIndex), "UTF-8");
        writer.println(model.layers.get(layerIndex));
        writer.println("Total Parameters{" + (weights.size + biases.size) + "}");
        writer.println("Number of Nodes{" + NUM_NODES + "}\n");
        writer.println("Number of Inputs{" + (inputs.size) + "}");
        writer.println(biases.toString(false) + "\n");
        writer.println("Number of weights{" + inputs.size + ", " + NUM_NODES + "}\n");
        writer.println(weights.toString(false));
        writer.close();
    }
}