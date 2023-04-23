import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class MaxPool extends Layer {
    final int kernelSize;
    INDArray[] gradients = new INDArray[Main.numThreads];

    public MaxPool(int kernelSize) {
        this.kernelSize = kernelSize;
    }

    @Override
    public INDArray forward(INDArray input, int threadIndex, int batchIndexForThread) {
        Matrix inputs = Matrix.convertToMatrix(input);
        Matrix grad = new Matrix(inputs.z, inputs.rows, inputs.cols);
        int resultSize = (int) Math.ceil((float) input.size(1) / (float) kernelSize);
        Matrix result = new Matrix((int)input.size(0), resultSize, resultSize);
        for(int l = 0; l < input.size(0); l++){
            for(int i = 0; i < input.size(1); i += kernelSize){
                for(int j = 0; j < input.size(2); j += kernelSize){
                    double maxVal = inputs.matrix[l][inputs.convert(i, j)];
                    int maxK = 0;
                    int maxH = 0;
                    for(int k = 0; k < kernelSize && i + k < inputs.rows; k++){
                        for(int h = 0; h < kernelSize && j + h < inputs.cols; h++){
                            if(inputs.matrix[l][inputs.convert(i + k, j + h)] > maxVal){
                                maxVal = inputs.matrix[l][inputs.convert(i + k, j + h)];
                                maxK = k;
                                maxH = h;
                            }
                            grad.matrix[l][inputs.convert(i + k, j + h)] = 0.0;
                        }
                    }
                    result.matrix[l][result.convert(i / kernelSize, j / kernelSize)] = maxVal;
                    grad.matrix[l][inputs.convert(i + maxK, j + maxH)] = 1.0;
                }
            }
        }
        gradients[threadIndex] = grad.convertToTensor();
        return result.convertToTensor();
    }

    @Override
    public INDArray backward(INDArray chain, int threadIndex) {
        Matrix prevGradients = Matrix.convertToMatrix(chain);
        Matrix grad = Matrix.convertToMatrix(gradients[threadIndex]);
        int i = 0;
        for(int j = 0; j < gradients[threadIndex].size(0); j++){
            for(int k = 0; k < gradients[threadIndex].size(1) * gradients[threadIndex].size(2); k++){

                grad.matrix[j][k] = grad.matrix[j][k] * prevGradients.matrix[j][prevGradients.convert(k / (int)gradients[threadIndex].size(2) / kernelSize,
                        (k % (int)gradients[threadIndex].size(2)) / kernelSize)];
                i++;
                if(i == kernelSize){
                    i = 0;
                }
            }
        }
        gradients[threadIndex] = grad.convertToTensor();
        return gradients[threadIndex];
    }

    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.println("Size of Kernals{" + kernelSize + "}");
        writer.close();
    }
}