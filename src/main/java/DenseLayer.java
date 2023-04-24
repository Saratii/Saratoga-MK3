import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;

public class DenseLayer extends Layer {
    public INDArray weightssss;
    private final INDArray bias;
    private final INDArray[] inputs = new INDArray[Main.numThreads];
    private final INDArray[] outputs = new INDArray[Main.numThreads];;
    int numOutChannels;
    private final INDArray[] biasGradients = new INDArray[Main.numThreads];
    private final INDArray[] weightGradients = new INDArray[Main.numThreads];
    public DenseLayer(int numInChannels, int numOutChannels) {
        this.numOutChannels = numOutChannels;
        for(int i = 0; i < weightGradients.length; i++){
            weightGradients[i] = Nd4j.zeros(DataType.DOUBLE, numInChannels, numOutChannels);
        }
        for(int i = 0; i < Main.numThreads; i++){
            biasGradients[i] = Nd4j.zeros(DataType.DOUBLE, 1, numOutChannels);
            outputs[i] = Nd4j.create(numOutChannels, 1);
        }
        bias = Nd4j.zeros(DataType.DOUBLE, 1, numOutChannels);
        Matrix weights = new Matrix(1, numInChannels, numOutChannels);
        weights.seedUniform();
        weightssss = weights.convertToTensor();
    }

    @Override
    public INDArray forward(INDArray input, int threadIndex, int batchIndexForThread) throws Exception {
        this.inputs[threadIndex] = input;
        INDArray weight = weightssss.reshape(weightssss.size(1), weightssss.size(0)).transpose();
        long[] inputDims = input.shape();
        if (inputDims.length != 2) {
            throw new Exception("flatten your input ");
        }
        if (inputDims[0] != weight.size(0)) {
            throw new Exception("invalid input channels in dense layer, expected " + inputDims[0]);
        }
        INDArray result = input.transpose().mmul(weight).add(bias);
        outputs[threadIndex] = result.transpose();
        return outputs[threadIndex];
    }

    @Override
    public INDArray backward(INDArray chain, int threadIndex) {
        chain = chain.reshape(chain.size(1), chain.size(0));
        biasGradients[threadIndex].addi(chain);
        INDArray passedOnDerivatives = Nd4j.create(DataType.DOUBLE,inputs[threadIndex].length(), 1);
        for(int i = 0; i < inputs[threadIndex].length(); i++){
            passedOnDerivatives.putScalar(i, 0.0);
            for(int j = 0; j < numOutChannels; j++){
                passedOnDerivatives.putScalar(i, passedOnDerivatives.getDouble(i) + chain.getDouble(j) * weightssss.getDouble(j * inputs[threadIndex].length() + i));
                weightGradients[threadIndex].putScalar(j * inputs[threadIndex].length() + i, weightGradients[threadIndex].getDouble(j * inputs[threadIndex].length() + i) + chain.getDouble(j) * inputs[threadIndex].getDouble(i));
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
                weightssss.putScalar(i, weightssss.getDouble(i) - weightGradients[j].getDouble(i) * Main.ALPHA);
            }
            weightGradients[j].assign(0.0);
        }
    }


    @Override
    public void write() throws IOException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, StandardCharsets.UTF_8);
        writer.println(this);
        writer.println("Total Parameters{" + (weightssss.length() + bias.length()) + "}");
        writer.println("Number of Nodes{" + numOutChannels + "}\n");
        writer.println("Number of Inputs{" + (inputs[0].length()) + "}");
        writer.println(bias.toString() + "\n");
        writer.println("Number of weights{" + inputs[0].length() + ", " + numOutChannels + "}\n");
        writer.println(weightssss.toString());
        writer.close();
    }
}