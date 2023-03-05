package src;

import static org.junit.Assert.assertThrows;

public class DenseLayer extends Layer{
    public Matrix weights;
    Matrix inputs;
    public Matrix outputs;
    int NUM_NODES;
    public Boolean initialized = false;
    public Matrix biases;
    Matrix batchWeightGradients;
    Matrix batchBiasGradients;
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
        Matrix passedOnDerivatives = new Matrix(1, inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            passedOnDerivatives.matrix[0][i] = 0.0;
            for(int j = 0; j < NUM_NODES; j++){
                passedOnDerivatives.matrix[0][i] += previousDerivatives.matrix[0][j] * weights.matrix[0][j * inputs.size + i];
                //weights.matrix[0][j * inputs.size + i] -= previousDerivatives.matrix[0][j] * inputs.matrix[0][i] * Main.ALPHA;
            }
        }
        Matrix tempy = previousDerivatives.multiply(inputs);
        if(firstBatch){
            batchBiasGradients = new Matrix(1, NUM_NODES, 1);
            batchWeightGradients = new Matrix(1, tempy.rows, tempy.cols);
            firstBatch = false;
        }
        batchBiasGradients.add(previousDerivatives);
        batchWeightGradients.add(tempy);

        return passedOnDerivatives;
    }
    public void updateParams(){
        firstBatch = true;
        for(int i = 0; i < batchBiasGradients.rows * batchBiasGradients.cols; i++){
            biases.matrix[0][i] -= batchBiasGradients.matrix[0][i] * Main.ALPHA / Main.BATCHSIZE;
        }
        for(int i = 0; i < batchWeightGradients.rows * batchWeightGradients.cols; i++){
            weights.matrix[0][i] -= batchWeightGradients.matrix[0][i] * Main.ALPHA / Main.BATCHSIZE;
        }
    }
}