
public class DenseLayer {
    Matrix weights;
    Matrix inputs;
    Matrix outputs;
    int NUM_NODES;
    Matrix biases;
    public DenseLayer(int NUM_NODES, Matrix inputs) {
        weights = new Matrix(inputs.size * NUM_NODES, 1);
        weights.seed();
        this.inputs = inputs;
        outputs = new Matrix(NUM_NODES, 1);
        this.NUM_NODES = NUM_NODES;
        Matrix biases = new Matrix(NUM_NODES, 1);
        biases.seed();
    }
    public Matrix forward(){
        for(int i = 0; i < NUM_NODES; i++){
            double sum = biases.matrix[i];
            for(int j = 0; j < inputs.size; j++){
                sum += inputs.matrix[j] * weights.matrix[i * inputs.size + j];
            }
            outputs.matrix[i] = sum;
        }
        return outputs;
    }
}
