import java.util.Random;

public class Node {
    Matrix inputs;
    int size;
    Matrix weights;
    double bias;
    public Node(Double[] inputs){
        this.size = inputs.length;
        this.inputs = new Matrix(size, 1);
        this.inputs.matrix = inputs;
        this.weights = new Matrix(size, 1);
        this.weights.seed();
        Random r = new Random();
        this.bias = r.nextDouble() * 2 - 1;
        
    }
    public double forward(){
        for(int i = 0; i < size; i++){
            bias += inputs.matrix[i] * weights.matrix[i];
        }
        return bias;
    }
}
