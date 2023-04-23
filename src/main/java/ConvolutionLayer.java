import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Random;

public class ConvolutionLayer extends Layer {
    INDArray[] input = new INDArray[Main.numThreads];
    Random r  = Main.r;
    final int NUM_IN_CHANNELS;
    final int NUM_OUT_CHANNELS;
    final int STRIDE;
    final int KERNAL_SIZE;
    INDArray[] biasGradients = new INDArray[Main.numThreads];
    Matrix[][] weightGradientsPerThread;
    public INDArray[] kernels;
    public INDArray bias;
    public ConvolutionLayer(int NUM_IN_CHANNELS, int NUM_OUT_CHANNELS, int STRIDE, int KERNAL_SIZE) {
        for(int i = 0; i < Main.numThreads; i++){
            biasGradients[i] = Nd4j.zeros(DataType.DOUBLE, NUM_OUT_CHANNELS, 1);
        }
        weightGradientsPerThread = new Matrix[Main.numThreads][NUM_IN_CHANNELS * NUM_OUT_CHANNELS];
        for(int i = 0; i < Main.numThreads; i++){
            for(int j = 0; j < weightGradientsPerThread[i].length; j++){
                weightGradientsPerThread[i][j] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE);
                weightGradientsPerThread[i][j].seedZeros();
            }
        }
        this.NUM_IN_CHANNELS = NUM_IN_CHANNELS;
        this.NUM_OUT_CHANNELS = NUM_OUT_CHANNELS;
        this.STRIDE = STRIDE;
        this.KERNAL_SIZE = KERNAL_SIZE;
        kernels = new INDArray[NUM_IN_CHANNELS * NUM_OUT_CHANNELS];
        bias = Nd4j.zeros(DataType.DOUBLE, NUM_OUT_CHANNELS, 1);
        for(int i = 0; i < kernels.length; i++){
            kernels[i] = Nd4j.create(KERNAL_SIZE, KERNAL_SIZE);
            for(int j = 0; j < kernels[i].length(); j++){
                kernels[i].putScalar(j, r.nextDouble() * 2 - 1);
            }
        }
    }

    @Override
    public INDArray forward(INDArray input, int threadIndex, int batchIndexForThread) {
        this.input[threadIndex] = input;
        Matrix result;
        if(input.shape().length > 2){
            result = new Matrix(NUM_OUT_CHANNELS, (int)input.size(1) - KERNAL_SIZE + 1, (int)input.size(2) - KERNAL_SIZE + 1);
        } else {
            result = new Matrix(NUM_OUT_CHANNELS, (int)input.size(0) - KERNAL_SIZE + 1, (int)input.size(1) - KERNAL_SIZE + 1);
        }

        result.seedZeros();
        for(int j = 0; j < NUM_OUT_CHANNELS; j++){
            for(int inputIndex = 0; inputIndex < NUM_IN_CHANNELS; inputIndex++){
                Double[] temp = Matrix.simpleCov(Matrix.convertToMatrix(input).matrix[inputIndex], Matrix.convertToMatrix(kernels[j * NUM_IN_CHANNELS + inputIndex]).matrix[0]);
                for(int k = 0; k < result.matrix[j].length; k++){
                    result.matrix[j][k] += temp[k];
                }
            }
            for(int k = 0; k < result.matrix[j].length; k++){
                result.matrix[j][k] += bias.getDouble(j);
            }
        }
        return result.convertToTensor();
    }

    @Override
    public Matrix backward(Matrix previousGradients, int threadIndex) {
        Matrix dldf = Matrix.convertToMatrix(input[threadIndex]).convolution(previousGradients);
        Matrix result;
        if(input[threadIndex].shape().length > 2) {
            result = new Matrix((int) input[threadIndex].size(0), (int) input[threadIndex].size(1), (int) input[threadIndex].size(2));
        } else {
            result = new Matrix(1, (int) input[threadIndex].size(0), (int) input[threadIndex].size(1));
        }
        Matrix biasGradient = new Matrix(1, NUM_OUT_CHANNELS, 1);
        Matrix[] kernalGradient = new Matrix[NUM_IN_CHANNELS * NUM_OUT_CHANNELS];
        for(int i = 0; i < kernels.length; i++){
            kernalGradient[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE);
        }
        result.seedZeros();
        for(int i = 0; i < result.z; i++){
            for(int j = 0; j < previousGradients.z; j++){
                Matrix temp = Matrix.convertToMatrix(kernels[j * result.z + i]).reverse().doubleBigConvolution(previousGradients.matrix[j], previousGradients.rows, previousGradients.cols);
                for(int k = 0; k < temp.innerSize; k++){
                    result.matrix[i][k] += temp.matrix[0][k];
                }
                for(int k = 0; k < kernels[j * result.z + i].length(); k++){
                    kernalGradient[j * result.z + i].matrix[0][k] = dldf.matrix[j * result.z + i][k];
                }
            }
        }
        for(int i = 0; i < kernalGradient.length; i++){
            weightGradientsPerThread[threadIndex][i].add(kernalGradient[i]);
        }
        for(int i = 0; i < previousGradients.z; i++){
            for(int j = 0; j < previousGradients.matrix[i].length; j++){
                biasGradient.matrix[0][i] = previousGradients.matrix[i][j];
            }
        }
        biasGradients[threadIndex] = biasGradients[threadIndex].add(biasGradient.convertToTensor());

        return result;
    }

    @Override
    public void updateParams() {
        for(int k = 0; k < weightGradientsPerThread.length; k++){
            for(int i = 0; i < weightGradientsPerThread[k].length; i++){
                for(int j = 0; j < weightGradientsPerThread[k][i].size; j++){
                    kernels[i].putScalar(j, kernels[i].getDouble(j) - weightGradientsPerThread[k][i].matrix[0][j] * Main.ALPHA);
                }
                weightGradientsPerThread[k][i].seedZeros();
            }
        }
        for(int i = 0; i < biasGradients.length; i++){
            for(int j = 0; j < biasGradients[i].length(); j++){
                bias.putScalar(j, bias.getDouble(j) - biasGradients[i].getDouble(j) * Main.ALPHA);
            }
            biasGradients[i].assign(0.0);
        }
    }

    @Override
    public void write() throws IOException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, StandardCharsets.UTF_8);
        writer.println(this);
        writer.println("Total Parameters{" + (kernels.length * kernels[0].length()) + "}");
        writer.println("Number of Kernals{" + kernels.length + "}");
        writer.println("Stride {" + STRIDE + "}");
        writer.println("Size of Kernals{" + KERNAL_SIZE + ", " + KERNAL_SIZE + "}\n");
        writer.println("Number of Input Channels{" + NUM_IN_CHANNELS + "}\n");
        writer.println("Number of Output Channels{" + NUM_OUT_CHANNELS + "}\n");
        writer.println("Biases{" + bias.length() + "}");
        writer.println(bias.toString() + "\n");
        writer.println("Kernal Weights{" + kernels.length * kernels[0].length() + "}");
        for(INDArray kernel : kernels){
            writer.println(kernel.toString());
        }
        writer.close();
    }
}
//Epoch: 0 Average Loss: 0.7261382523683187
//Percentage Correct: NaN
//Total Correct: 0 out of: 0
//Epoch: 1 Average Loss: 0.5614891069143072
//Percentage Correct: NaN
//Total Correct: 0 out of: 0
//Epoch: 2 Average Loss: 0.5015242956784484