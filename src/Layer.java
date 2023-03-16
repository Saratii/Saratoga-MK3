package src;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;

public class Layer {
    public Matrix forward(Matrix inputs){
        return inputs;
    }
    public Matrix backward(Matrix inputs){
        return inputs;
    }
    public void updateParams(){
        
    }
    public void write() throws FileNotFoundException, UnsupportedEncodingException{

    }
   
}
