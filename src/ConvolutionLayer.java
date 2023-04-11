package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

public class ConvolutionLayer extends Layer{
    Matrix input;
    int NUM_IN_CHANNELS;
    int NUM_OUT_CHANNELS;
    int STRIDE;
    int KERNAL_SIZE;
    List<Matrix> biasGradientsPerThread = new ArrayList<>();
    List<Matrix[]> weightGradientsPerThread = new ArrayList<>();
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
            
            biases = new Matrix(1, NUM_OUT_CHANNELS, 1);
            biases.seedZeros();
            
            for(int i = 0; i < kernals.length; i++){
                kernals[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE); 
                kernals[i].seed();
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
        Matrix biasGradient = new Matrix(1, NUM_OUT_CHANNELS, 1);
        Matrix[] kernalGradient = new Matrix[NUM_IN_CHANNELS * NUM_OUT_CHANNELS];
        for(int i = 0; i < kernals.length; i++){
            kernalGradient[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE);
        }
        result.seedZeros();
        for(int i = 0; i < result.z; i++){
            for(int j = 0; j < previousGradients.z; j++){
                Matrix temp = kernals[j * result.z + i].reverse().doubleBigConvolution(previousGradients.matrix[j], previousGradients.rows, previousGradients.cols);
                for(int k = 0; k < temp.rows * temp.cols; k++){
                    result.matrix[i][k] += temp.matrix[0][k];
                }
                for(int k = 0; k < kernals[j * result.z + i].size; k++){
                    kernalGradient[j * result.z + i].matrix[0][k] = dldf.matrix[j * result.z + i][k]; 
                }
            }
        }
        weightGradientsPerThread.add(kernalGradient);
        for(int i = 0; i < previousGradients.z; i++){
            for(int j = 0; j < previousGradients.matrix[i].length; j++){
                biasGradient.matrix[0][i] = previousGradients.matrix[i][j];
            }
        }
        biasGradientsPerThread.add(biasGradient);
        return result;
    }
    public void updateParams(){
        for(int k = 0; k < weightGradientsPerThread.size(); k++){
            for(int i = 0; i < weightGradientsPerThread.get(k).length; i++){
                for(int j = 0; j < weightGradientsPerThread.get(k)[i].size; j++){
                    kernals[i].matrix[0][j] -= weightGradientsPerThread.get(k)[i].matrix[0][j] * Main.ALPHA;
                }
            }
        }
        for(int i = 0; i < biasGradientsPerThread.size(); i++){
            for(int j = 0; j < biasGradientsPerThread.get(i).size; j++){
                biases.matrix[0][j] -= biasGradientsPerThread.get(i).matrix[0][j] * Main.ALPHA;
            }
        }
    }
    public void write(int layerIndex) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("logs/log-" +  Main.model.layers.get(layerIndex), "UTF-8");
        writer.println(Main.model.layers.get(layerIndex));
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