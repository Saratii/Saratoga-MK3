
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
        inputs = outputs;
        return outputs;
    }
    public Matrix softmax(){
        Matrix results = new Matrix(inputs.size, 1);
        double sum = 0;
        for( int i = 0; i < inputs.size; i++){
            results.matrix[i] = Math.exp(inputs.matrix[i]);
            sum += results.matrix[i];
        }
        for(int i = 0; i < results.size; i++){
            results.matrix[i] /= sum;
        }
        return results;
    }
}
