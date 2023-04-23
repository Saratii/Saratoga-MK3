import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class DenseLayer extends Layer {
    public Matrix weights;
    private INDArray bias;
    private INDArray[] inputs = new INDArray[Main.numThreads];
    private INDArray[] outputs = new INDArray[Main.numThreads];;
    int numOutChannels;
    private final INDArray[] biasGradients = new INDArray[Main.numThreads];
    private final INDArray[] weightGradients = new INDArray[Main.numThreads];
    public DenseLayer(int numInChannels, int numOutChannels) {
        this.numOutChannels = numOutChannels;
        for(int i = 0; i < weightGradients.length; i++){
            weightGradients[i] = Nd4j.zeros(DataType.DOUBLE, numInChannels, numOutChannels);
        }
        for(int i = 0; i < Main.numThreads; i++){
            biasGradients[i] = Nd4j.zeros(DataType.DOUBLE, numOutChannels, 1);
            outputs[i] = Nd4j.create(numOutChannels, 1);
        }
        bias = Nd4j.zeros(DataType.DOUBLE, numOutChannels, 1);
        weights = new Matrix(1, numInChannels, numOutChannels);
        weights.seedUniform();
    }

    @Override
    public INDArray forward(INDArray input, int threadIndex, int batchIndexForThread) throws Exception {
        this.inputs[threadIndex] = input;
        INDArray weight = weights.convertToTensor();
        weight = weight.reshape(weight.size(1), weight.size(0)).transpose();
        long[] inputDims = input.shape();
        long[] weightDims = weight.shape();
        if (inputDims.length != 2) {
            throw new Exception("flatten your input ");
        }
        if (inputDims[0] != weightDims[0]) {
            throw new Exception("invalid input channels in dense layer, expected " + inputDims[0]);
        }
        INDArray result = input.transpose().mmul(weight).add(bias.reshape(bias.size(1), bias.size(0)));
        outputs[threadIndex] = result.transpose();
        return outputs[threadIndex];
    }

    @Override
    public Matrix backward(Matrix chain, int threadIndex) {
        biasGradients[threadIndex].addi(chain.convertToTensor());
        Matrix passedOnDerivatives = new Matrix(1, (int) inputs[threadIndex].length(), 1);
        for(int i = 0; i < inputs[threadIndex].length(); i++){
            passedOnDerivatives.matrix[0][i] = 0.0;
            for(int j = 0; j < numOutChannels; j++){
                passedOnDerivatives.matrix[0][i] += chain.matrix[0][j] * weights.matrix[0][j * (int)inputs[threadIndex].length() + i];
                weightGradients[threadIndex].putScalar(j * inputs[threadIndex].length() + i, weightGradients[threadIndex].getDouble(j * inputs[threadIndex].length() + i) + chain.matrix[0][j] * inputs[threadIndex].getDouble(i));
            }
        }
        return passedOnDerivatives;
    }

    @Override
    public void updateParams() {
        for(int j = 0; j < biasGradients.length; j++) {
            bias.subi(biasGradients[j].mul(Main.ALPHA));
            biasGradients[j].assign(0.0);
        }
        for(int j = 0; j < weightGradients.length; j++){
            for(int i = 0; i < weightGradients[j].length(); i++){
                weights.matrix[0][i] -= weightGradients[j].getDouble(i) * Main.ALPHA;
            }
            weightGradients[j].assign(0.0);
        }
    }


    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.println("Total Parameters{" + (weights.size + bias.length()) + "}");
        writer.println("Number of Nodes{" + numOutChannels + "}\n");
        writer.println("Number of Inputs{" + (inputs[0].length()) + "}");
        writer.println(bias.toString() + "\n");
        writer.println("Number of weights{" + inputs[0].length() + ", " + numOutChannels + "}\n");
        writer.println(weights.toString(false));
        writer.close();
    }
}