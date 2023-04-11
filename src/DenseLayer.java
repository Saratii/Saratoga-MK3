package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DenseLayer extends Layer{
    public Matrix weights;
    Matrix inputs;
    Matrix outputs;
    int NUM_NODES;
    Boolean initialized = false;
    public Matrix biases;
    List<Matrix> weightGradientsPerThread = new ArrayList<>();
    List<Matrix> biasGradientsPerThread = new ArrayList<>();
    public DenseLayer(int NUM_NODES) {
        this.NUM_NODES = NUM_NODES;
    }
    public Matrix forward(Matrix inputs){
        if(!initialized){
            biases = new Matrix(1, NUM_NODES, 1);
            biases.seedZeros();
            weights = new Matrix(1, inputs.size, NUM_NODES);
            weights.seedUniform();
            outputs = new Matrix(1, NUM_NODES, 1);
            initialized = true;
        }
        this.inputs = inputs;
        for(int i = 0; i < NUM_NODES; i++){
            double sum = biases.matrix[0][i];
            for(int j = 0; j < inputs.rows * inputs.cols; j++){
                sum += inputs.matrix[0][j] * weights.matrix[0][i * inputs.rows * inputs.cols + j];
            }
            outputs.matrix[0][i] = sum;
        }
        inputs = outputs;
        return outputs;
    }
    public Matrix backward(Matrix previousDerivatives){
        biasGradientsPerThread.add(previousDerivatives);
        Matrix weightGradient = new Matrix(1, inputs.size, NUM_NODES);
        Matrix passedOnDerivatives = new Matrix(1, inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            passedOnDerivatives.matrix[0][i] = 0.0;
            for(int j = 0; j < NUM_NODES; j++){
                passedOnDerivatives.matrix[0][i] += previousDerivatives.matrix[0][j] * weights.matrix[0][j * inputs.size + i];
                weightGradient.matrix[0][j * inputs.size + i] = previousDerivatives.matrix[0][j] * inputs.matrix[0][i];
            }
        }
        weightGradientsPerThread.add(weightGradient);
        return passedOnDerivatives;
    }
    public void updateParams(){
        for(int j = 0; j < biasGradientsPerThread.size(); j++){
            for(int i = 0; i < biasGradientsPerThread.get(j).size; i++){
                biases.matrix[0][i] -= biasGradientsPerThread.get(j).matrix[0][i] * Main.ALPHA;
            }
        }
        for(int j = 0; j < weightGradientsPerThread.size(); j++){
            for(int i = 0; i < weightGradientsPerThread.get(j).size; i++){
                weights.matrix[0][i] -= weightGradientsPerThread.get(j).matrix[0][i] * Main.ALPHA;
            }
        }
    }
    public void write(int layerIndex) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("logs/log-" +  Main.model.layers.get(layerIndex), "UTF-8");
        writer.println(Main.model.layers.get(layerIndex));
        writer.println("Total Parameters{" + (weights.size + biases.size) + "}");
        writer.println("Number of Nodes{" + NUM_NODES + "}\n");
        writer.println("Number of Inputs{" + (inputs.size) + "}");
        writer.println(biases.toString(false) + "\n");
        writer.println("Number of weights{" + inputs.size + ", " + NUM_NODES + "}\n");
        writer.println(weights.toString(false));
        writer.close();
    }
}