import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class MaxPool extends Layer {
    final int kernalSize;
    INDArray[] gradients = new INDArray[Main.numThreads];

    public MaxPool(int kernelSize) {
        this.kernalSize = kernelSize;
    }

    @Override
    public INDArray forward(INDArray input, int threadIndex, int batchIndexForThread) {
        Matrix inputs = Matrix.convertToMatrix(input);
        Matrix grad = new Matrix(inputs.z, inputs.rows, inputs.cols);
        int resultSize = (int) Math.ceil((float) input.size(1) / (float) kernalSize);
        Matrix result = new Matrix((int)input.size(0), resultSize, resultSize);
        for(int l = 0; l < input.size(0); l++){
            for(int i = 0; i < input.size(1); i += kernalSize){
                for(int j = 0; j < input.size(2); j += kernalSize){
                    double maxVal = inputs.matrix[l][inputs.convert(i, j)];
                    int maxK = 0;
                    int maxH = 0;
                    for(int k = 0; k < kernalSize && i + k < inputs.rows; k++){
                        for(int h = 0; h < kernalSize && j + h < inputs.cols; h++){
                            if(inputs.matrix[l][inputs.convert(i + k, j + h)] > maxVal){
                                maxVal = inputs.matrix[l][inputs.convert(i + k, j + h)];
                                maxK = k;
                                maxH = h;
                            }
                            grad.matrix[l][inputs.convert(i + k, j + h)] = 0.0;
                        }
                    }
                    result.matrix[l][result.convert(i / kernalSize, j / kernalSize)] = maxVal;
                    grad.matrix[l][inputs.convert(i + maxK, j + maxH)] = 1.0;
                }
            }
        }
        gradients[threadIndex] = grad.convertToTensor();
        return result.convertToTensor();
    }

    @Override
    public Matrix backward(Matrix prevGradients, int threadIndex) {
        INDArray chain = prevGradients.convertToTensor();
        Matrix grad = Matrix.convertToMatrix(gradients[threadIndex]);
        int i = 0;
        for(int j = 0; j < grad.z; j++){
            for(int k = 0; k < grad.innerSize; k++){
                grad.matrix[j][k] *= prevGradients.matrix[j][prevGradients.convert(k / grad.cols / kernalSize,
                        (k % grad.cols) / kernalSize)];
                i++;
                if(i == kernalSize){
                    i = 0;
                }
            }
        }
        gradients[threadIndex] = grad.convertToTensor();
        return grad;
    }

    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.println("Size of Kernals{" + kernalSize + "}");
        writer.close();
    }
}