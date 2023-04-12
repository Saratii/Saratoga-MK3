package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;

public class Softmax extends Layer{
    Matrix layerOutput;
    Matrix result;
    public Matrix forward(Matrix inputs){
        result = new Matrix(1, inputs.size, 1);
        for(int i = 0; i < inputs.size; i++){
            double shiftConstant = Matrix.maxValue(inputs.matrix[0]);
            double sum = 0.0;
            for(int j = 0; j < inputs.size; j++){
                sum+= Math.exp(inputs.matrix[0][j] - shiftConstant);
            }
           
            result.matrix[0][i] = Math.exp(inputs.matrix[0][i] - shiftConstant) / sum;
            if(Double.isNaN(result.matrix[0][i])){
                System.out.println("mommu did a fucky wucky");
                Arrays.toString(inputs.matrix[0]);
            }
        }
        layerOutput = result;
        return result;
    }
    public Matrix backward(Matrix dvalues){
        Matrix result = new Matrix(1, dvalues.size, dvalues.size);
        for(int i = 0; i < dvalues.size; i++){
            for(int j = 0; j < dvalues.size; j++){
                result.matrix[0][j*dvalues.size + i] = layerOutput.matrix[0][i] * ((i == j ? 1 : 0) - layerOutput.matrix[0][j]);
                if(Double.isNaN(result.matrix[0][j*dvalues.size + i])){
                    System.out.println("mommu did a fucky wucky");
                    Arrays.toString(layerOutput.matrix[0]);
                }
                
            }
        }
        return result.multiply(dvalues);
    }
    public void write(int layerIndex, Model model) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("logs/log-" +  model.layers.get(layerIndex), "UTF-8");
        writer.println(model.layers.get(layerIndex));
        writer.close();
    }
}