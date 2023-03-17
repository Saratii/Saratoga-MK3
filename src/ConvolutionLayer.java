package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Random;

public class ConvolutionLayer extends Layer{
    Matrix input;
    int NUM_FEATURE_SETS;
    int STRIDE;
    int KERNAL_SIZE;
    private Matrix[] kernalGradient;
    public Matrix[] kernals;
    public boolean initialized = false;
   
    public ConvolutionLayer(int NUM_FEATURE_SETS, int STRIDE, int KERNAL_SIZE){
        this.NUM_FEATURE_SETS = NUM_FEATURE_SETS;
        this.STRIDE = STRIDE;
        this.KERNAL_SIZE = KERNAL_SIZE;
    }
    
    public Matrix forward(Matrix input){
        this.input = input;
        if(!initialized){
            this.kernals = new Matrix[NUM_FEATURE_SETS];
            kernalGradient = new Matrix[NUM_FEATURE_SETS];
            for(int i = 0; i < kernals.length; i++){
                kernals[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE); 
                kernalGradient[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE);
                kernals[i].seed();
                kernalGradient[i].seedZeros();
            }
            initialized = true;
        }
        // Matrix result = new Matrix(NUM_FEATURE_SETS * input.z, input.rows - KERNAL_SIZE + 1, input.cols - KERNAL_SIZE + 1);
        // for(int i = 0 ; i < result.matrix.length; i++){
        //     result.matrix[i] = input.convolution(kernals[i % kernals.length]).matrix[0];
        // }
        Matrix result = new Matrix(NUM_FEATURE_SETS, input.rows - KERNAL_SIZE + 1, input.cols - KERNAL_SIZE + 1);
        result.seedZeros();
        for(int j = 0; j < NUM_FEATURE_SETS; j++){
            for(int i = 0; i < input.z; i++){
                Double[] temp = Matrix.simpleCov(input.matrix[i], kernals[j].matrix[0]);
                for(int k = 0; k < result.matrix[j].length; k++){
                    result.matrix[j][k] += temp[k];
                }
            }
            for(int k = 0; k < result.matrix[j].length; k++){
                result.matrix[j][k] /= input.z;
            }
        }
        return result;
    }
    public Matrix backward(Matrix previousGradients){ //dLdO = previousGradients
        Matrix dldf = input.convolution(previousGradients);
        Matrix result = new Matrix(previousGradients.z * kernals.length, input.rows, input.cols);
        for(int i = 0; i < kernals.length; i++){
            Matrix temp = kernals[i];
            temp.reverse(); //i just need to know how to fix this file not optimize anything
            Matrix dldx = temp.bigConvolution(previousGradients);
            for(int h = 0; h < dldx.z; h++){
                for(int k = 0; k < dldx.rows * dldx.cols; k++){
                    result.matrix[i * previousGradients.z + h][k] = dldx.matrix[h][k];
                }
            }
            for(int j = 0; j < kernals[i].size; j++){
                kernalGradient[i].matrix[0][j] += dldf.matrix[i][j];
            }
        }
        return result;
    }
    public void updateParams(){
        for(int i = 0; i < kernalGradient.length; i++){
            for(int j = 0; j < kernalGradient[i].size; j++){
                kernals[i].matrix[0][j] -= kernalGradient[i].matrix[0][j] * Main.ALPHA / Main.batchSize;
            }
        }
    }
    public void write() throws FileNotFoundException, UnsupportedEncodingException{
        Random r = new Random();
        int k = r.nextInt(10000, 99999);
        PrintWriter writer = new PrintWriter("logs/log-Conv.txt" + k, "UTF-8");
        writer.println("Convolutional Layer");
        writer.println("Total Parameters{" + (kernals.length * kernals[0].size) + "}");
        writer.println("Number of Kernals{" + kernals.length + "}");
        writer.println("Size of Kernals{" +  kernals[0].rows + ", " + kernals[0].cols + "}\n");
        for(Matrix kernal : kernals){
            writer.println(kernal.toString());
            writer.println("");
        }
        writer.close();
    }
}