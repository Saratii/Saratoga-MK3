package src;
public class DenseLayer extends Layer{
    public Matrix weights;
    Matrix inputs;
    Matrix outputs;
    int NUM_NODES;
    public Matrix biases;
    public DenseLayer(int NUM_INPUTS, int NUM_NODES) {
        weights = new Matrix(NUM_INPUTS, NUM_NODES);
        weights.seedUniform();
        outputs = new Matrix(NUM_NODES, 1);
        this.NUM_NODES = NUM_NODES;
        biases = new Matrix(NUM_NODES, 1);
        biases.seedZeros();
    }
    public Matrix forward(Matrix inputs){
        this.inputs = inputs;
        for(int i = 0; i < NUM_NODES; i++){
            double sum = biases.matrix[i];
            for(int j = 0; j < inputs.size; j++){
                sum += inputs.matrix[j] * weights.matrix[i * inputs.size + j];
            }
            outputs.matrix[i] = sum;
        }
        inputs = outputs;
        return outputs;
    }
    public Matrix backward(Matrix previousDerivatives){
        for(int i = 0; i < previousDerivatives.size; i++){
            biases.matrix[i] -= previousDerivatives.matrix[i] * Main.ALPHA;
        }
        Matrix passedOnDerivatives = new Matrix(inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            passedOnDerivatives.matrix[i] = 0.0;
            for(int j = 0; j < NUM_NODES; j++){
                passedOnDerivatives.matrix[i] += previousDerivatives.matrix[j] * weights.matrix[j * inputs.size + i];
                weights.matrix[j * inputs.size + i] -= previousDerivatives.matrix[j] * inputs.matrix[i] * Main.ALPHA;
            }
        }
        return passedOnDerivatives;
    }
}
