package src;
public class DenseLayer extends Layer{
    public Matrix weights;
    Matrix inputs;
    Matrix outputs;
    int NUM_NODES;
    Boolean initialized = false;
    public Matrix biases;
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
        for(int i = 0; i < previousDerivatives.size; i++){
            biases.matrix[0][i] -= previousDerivatives.matrix[0][i] * Main.ALPHA;
        }
        Matrix passedOnDerivatives = new Matrix(1, inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            passedOnDerivatives.matrix[0][i] = 0.0;
            for(int j = 0; j < NUM_NODES; j++){
                passedOnDerivatives.matrix[0][i] += previousDerivatives.matrix[0][j] * weights.matrix[0][j * inputs.size + i];
                weights.matrix[0][j * inputs.size + i] -= previousDerivatives.matrix[0][j] * inputs.matrix[0][i] * Main.ALPHA;
            }
        }
        return passedOnDerivatives;
    }
}
