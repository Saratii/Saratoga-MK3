package src;
import java.util.ArrayList;

public class Model {
    ArrayList<Layer> layers = new ArrayList<>();
    Matrix inputs;
    Matrix expected;
    long startTime;
    public double forward(Matrix inputs, Matrix expected){
        System.out.println("\n");
        for(int i = 0; i < layers.size(); i++){
            startTime = System.currentTimeMillis();
            inputs = layers.get(i).forward(inputs);
            System.out.println(String.format("Finished Forward %s in %d ms", layers.get(i),  System.currentTimeMillis() - startTime));
        }
        double l = Loss.forward(inputs, expected);
        System.out.println("\nSoftmax output: " + inputs);
        System.out.println("Loss: " + l + "\n");
        this.expected = expected;
        this.inputs = inputs;
        return l;
    }
    public void backward(){
        Matrix gradients = Loss.backward(inputs, expected);
        for(int i = 0; i < layers.size(); i++){
            startTime = System.currentTimeMillis();
            gradients = layers.get(layers.size() - 1 - i).backward(gradients);
            System.out.println(String.format("Finished Backward %s in %d ms", layers.get(layers.size() - 1 - i),  System.currentTimeMillis() - startTime));
        }
    }
} //just a sec 