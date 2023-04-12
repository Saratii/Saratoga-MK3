package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class ConvolutionLayer extends Layer{
    Matrix input;
    int NUM_IN_CHANNELS;
    int NUM_OUT_CHANNELS;
    int STRIDE;
    int KERNAL_SIZE;
    private Matrix[] kernalGradient;
    Matrix biasGradient;
    public Matrix[] kernals;
    public Matrix biases;
    public boolean initialized = false;
   
    public ConvolutionLayer(int NUM_IN_CHANNELS, int NUM_OUT_CHANNELS, int STRIDE, int KERNAL_SIZE){
        this.NUM_IN_CHANNELS = NUM_IN_CHANNELS;
        this.NUM_OUT_CHANNELS = NUM_OUT_CHANNELS;
        this.STRIDE = STRIDE;
        this.KERNAL_SIZE = KERNAL_SIZE;
        if(!initialized){
            kernals = new Matrix[NUM_IN_CHANNELS * NUM_OUT_CHANNELS];
            kernalGradient = new Matrix[NUM_IN_CHANNELS * NUM_OUT_CHANNELS];
            biases = new Matrix(1, NUM_OUT_CHANNELS, 1);
            biasGradient = new Matrix(1, NUM_OUT_CHANNELS, 1);
            biases.seedZeros();
            biasGradient.seedZeros();
            for(int i = 0; i < kernals.length; i++){
                kernals[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE); 
                kernalGradient[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE);
                kernals[i].seed();
                kernalGradient[i].seedZeros();
            }
            initialized = true;
        }
    }
    
    public Matrix forward(Matrix input){
        this.input = input;
        Matrix result = new Matrix(NUM_OUT_CHANNELS, input.rows - KERNAL_SIZE + 1, input.cols - KERNAL_SIZE + 1);
        result.seedZeros();
        for(int j = 0; j < NUM_OUT_CHANNELS; j++){
            for(int inputIndex = 0; inputIndex < NUM_IN_CHANNELS; inputIndex++){
                Double[] temp = Matrix.simpleCov(input.matrix[inputIndex], kernals[j * NUM_IN_CHANNELS + inputIndex].matrix[0]);
                for(int k = 0; k < result.matrix[j].length; k++){
                    result.matrix[j][k] += temp[k];
                }
            }
            for(int k = 0; k < result.matrix[j].length; k++){
                result.matrix[j][k] += biases.matrix[0][j];
            }
        }
        return result;
    } //gimme call stack
    public Matrix backward(Matrix previousGradients){ //dLdO = previousGradients
        Matrix dldf = input.convolution(previousGradients);
        Matrix result = new Matrix(input.z, input.rows, input.cols);
        result.seedZeros();
        for(int i = 0; i < result.z; i++){
            for(int j = 0; j < previousGradients.z; j++){
                Matrix temp = kernals[j * result.z + i].reverse().doubleBigConvolution(previousGradients.matrix[j], previousGradients.rows, previousGradients.cols);
                for(int k = 0; k < temp.rows * temp.cols; k++){
                    result.matrix[i][k] += temp.matrix[0][k];
                }
                for(int k = 0; k < kernals[j * result.z + i].size; k++){
                    kernalGradient[j * result.z + i].matrix[0][k] += dldf.matrix[j * result.z + i][k]; 
                }
            }
        }
        
        for(int i = 0; i < previousGradients.z; i++){
            for(int j = 0; j < previousGradients.matrix[i].length; j++){
                biasGradient.matrix[0][i] += previousGradients.matrix[i][j];
            }
        }
        return result;
    }
    public void updateParams(){
        for(int i = 0; i < kernalGradient.length; i++){
            for(int j = 0; j < kernalGradient[i].size; j++){
                kernals[i].matrix[0][j] -= kernalGradient[i].matrix[0][j] * Main.ALPHA;
            }
        }
        for(int j = 0; j < biasGradient.size; j++){
            biases.matrix[0][j] -= biasGradient.matrix[0][j] * Main.ALPHA;
        }
        biasGradient.seedZeros();
        for(int k = 0; k < kernalGradient.length; k++){
            kernalGradient[k].seedZeros();
        }
    }
    public void write(int layerIndex, Model model) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("logs/log-" + model.layers.get(layerIndex), "UTF-8");
        writer.println(model.layers.get(layerIndex));
        writer.println("Total Parameters{" + (kernals.length * kernals[0].size) + "}");
        writer.println("Number of Kernals{" + kernals.length + "}");
        writer.println("Stride {" + STRIDE + "}");
        writer.println("Size of Kernals{" +  kernals[0].rows + ", " + kernals[0].cols + "}\n");
        writer.println("Number of Input Channels{" + NUM_IN_CHANNELS + "}\n");
        writer.println("Number of Output Channels{" + NUM_OUT_CHANNELS + "}\n");
        writer.println("Biases{" + biases.size + "}");
        writer.println(biases.toString(false) + "\n");
        writer.println("Kernal Weights{" + kernals.length * kernals[0].size + "}");
        for(Matrix kernal : kernals){
            writer.println(kernal.toString(false)); 
        }
        writer.close();
    }
    
}