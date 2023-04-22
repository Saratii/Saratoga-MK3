import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class DenseLayer extends Layer {
    public Matrix weights;
    public Matrix biases;
    public INDArray bias;
    Matrix[] inputs = new Matrix[Main.numThreads];
    Matrix[] outputs = new Matrix[Main.numThreads];;
    int numOutChannels;
    private final Matrix[] biasGradient = new Matrix[Main.numThreads];;
    private final Matrix[] weightGradient = new Matrix[Main.numThreads];
    public DenseLayer(int numInChannels, int numOutChannels) {
        this.numOutChannels = numOutChannels;
        for(int i = 0; i < weightGradient.length; i++){
            weightGradient[i] = new Matrix(1, numInChannels, numOutChannels);
            weightGradient[i].seedZeros();
        }
        for(int i = 0; i < Main.numThreads; i++){
            biasGradient[i] = new Matrix(1, numOutChannels, 1);
            biasGradient[i].seedZeros();
            outputs[i] = new Matrix(1, numOutChannels, 1);
        }
        biases = new Matrix(1, numOutChannels, 1);
        biases.seedZeros();
        bias = Nd4j.zeros(numOutChannels, 1);
        weights = new Matrix(1, numInChannels, numOutChannels);
        weights.seedUniform();
    }

    @Override
    public Matrix forward(Matrix inputs, int threadIndex, int batchIndexForThread) throws Exception {
        this.inputs[threadIndex] = inputs;
        INDArray input = inputs.convertToTensor();
        INDArray weight = weights.convertToTensor();
        weight = weight.reshape(weight.size(1), weight.size(0)).transpose();
        INDArray biase = biases.convertToTensor();
        long[] inputDims = input.shape();
        long[] weightDims = weight.shape();
        if (inputDims.length != 2) {
            throw new Exception("flatten your input ");
        }
        if (inputDims[0] != weightDims[0]) {
            throw new Exception("invalid input channels in dense layer, expected " + inputDims[0]);
        }
        INDArray result = input.transpose().mmul(weight).add(biase.reshape(bias.size(1), bias.size(0)));
        outputs[threadIndex] = Matrix.convertToMatrix(result.transpose());
        return outputs[threadIndex];
    }

//    Epoch: 4 Average Loss: 0.5719871986523103
//    Completed in 5 epochs
//    Average time per epoch: 2913 ms <-- mine


    @Override
    public Matrix backward(Matrix previousDerivatives, int threadIndex) {
        biasGradient[threadIndex] = Matrix.convertToMatrix(biasGradient[threadIndex].convertToTensor().add(previousDerivatives.convertToTensor()));
        Matrix passedOnDerivatives = new Matrix(1, inputs[threadIndex].size, 1);
        for(int i = 0; i < inputs[threadIndex].size; i++){
            passedOnDerivatives.matrix[0][i] = 0.0;
            for(int j = 0; j < numOutChannels; j++){
                passedOnDerivatives.matrix[0][i] += previousDerivatives.matrix[0][j] * weights.matrix[0][j * inputs[threadIndex].size + i];
                weightGradient[threadIndex].matrix[0][j * inputs[threadIndex].size + i] += previousDerivatives.matrix[0][j] * inputs[threadIndex].matrix[0][i];
            }
        }
        return passedOnDerivatives;
    }

    @Override
    public void updateParams() {
        for(int j = 0; j < biasGradient.length; j++) {
            INDArray biasGrad = biasGradient[j].convertToTensor();
            INDArray bias = biases.convertToTensor();
            biases = Matrix.convertToMatrix(bias.sub(biasGrad.mul(Main.ALPHA)));
            biasGradient[j].seedZeros();
        }
        for(int j = 0; j < weightGradient.length; j++){
            for(int i = 0; i < weightGradient[j].size; i++){
                weights.matrix[0][i] -= weightGradient[j].matrix[0][i] * Main.ALPHA;
            }
            weightGradient[j].seedZeros();
        }
    }

    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.println("Total Parameters{" + (weights.size + biases.size) + "}");
        writer.println("Number of Nodes{" + numOutChannels + "}\n");
        writer.println("Number of Inputs{" + (inputs[0].size) + "}");
        writer.println(biases.toString(false) + "\n");
        writer.println("Number of weights{" + inputs[0].size + ", " + numOutChannels + "}\n");
        writer.println(weights.toString(false));
        writer.close();
    }
}