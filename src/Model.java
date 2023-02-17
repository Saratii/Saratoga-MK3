package src;
import java.util.ArrayList;

public class Model {
    ArrayList<Layer> layers = new ArrayList<>();
    Matrix inputs;
    Matrix expected;
    public double forward(Matrix inputs, Matrix expected){
        for(int i = 0; i < layers.size(); i++){
            inputs = layers.get(i).forward(inputs);
        }
        double l = Loss.forward(inputs, expected);
        System.out.println("Softmax output: " + inputs);
        System.out.println("Loss: " + l + "\n");
        this.expected = expected;
        this.inputs = inputs;
        return l;
    }
    public void backward(){
        Matrix gradients = Loss.backward(inputs, expected);
        for(int i = 1; !(layers.get(layers.size() - i) instanceof Flatten); i++){
            gradients = layers.get(layers.size() - i).backward(gradients);
        }
    }
}