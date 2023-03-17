package src;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

public class Model {
    public ArrayList<Layer> layers = new ArrayList<>();
    Matrix inputs;
    Matrix expected;
    long startTime;
    Boolean profiling = false;
    public double forward(Matrix inputs, Matrix expected){
        if(profiling){
            // System.out.println("\n");
        }
        for(int i = 0; i < layers.size(); i++){
            startTime = System.currentTimeMillis();
            inputs = layers.get(i).forward(inputs);
            if(profiling){
                // System.out.println(String.format("Finished Forward %s in %d ms", layers.get(i),  System.currentTimeMillis() - startTime));
                layers.get(i).forwardTime += System.currentTimeMillis() - startTime;
            }
        }
        double l = Loss.forward(inputs, expected);
        this.expected = expected;
        this.inputs = inputs;
        return l;
    }
    public void backward(){
        Matrix gradients = Loss.backward(inputs, expected);
        for(int i = 0; i < layers.size(); i++){
            startTime = System.currentTimeMillis();
            gradients = layers.get(layers.size() - 1 - i).backward(gradients);
            if(profiling){
                // System.out.println(String.format("Finished Backward %s in %d ms", layers.get(layers.size() - 1 - i),  System.currentTimeMillis() - startTime));
                layers.get(layers.size() - 1 - i).backwardTime += System.currentTimeMillis() - startTime;
            }
        }
    }
    public void updateParams(){
        for(Layer layer: layers){
            layer.updateParams();
        }
    }
    public void write() throws FileNotFoundException, UnsupportedEncodingException{
        for(Layer layer: layers){
            layer.write();
        }
    }
}