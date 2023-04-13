package src;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

public class Model {
    public ArrayList<Layer> layers = new ArrayList<>();
    Matrix inputs;
    Matrix expected;
    long startTime;
    Boolean profiling = false;
    public double forward(Matrix inputs, Matrix expected, int threadIndex) throws Exception{
        for(int i = 0; i < layers.size(); i++){
            startTime = System.currentTimeMillis();
            inputs = layers.get(i).forward(inputs, threadIndex);
        }
        double l = Loss.forward(inputs, expected);
        this.expected = expected;
        this.inputs = inputs;
        return l;
    }
    public void backward(int threadIndex){
        Matrix gradients = Loss.backward(inputs, expected);
        for(int i = 0; i < layers.size(); i++){
            startTime = System.currentTimeMillis();
            gradients = layers.get(layers.size() - 1 - i).backward(gradients, threadIndex);
        }
    }
    public void updateParams(){
        for(Layer layer: layers){
            layer.updateParams();
        }
    }
    public void write(Model model) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("logs/log-architecture", "UTF-8");
        for(int i = 0; i < layers.size(); i++){
            layers.get(i).write(i, model);
            writer.println(layers.get(i));
        }
        writer.close();
    }
}