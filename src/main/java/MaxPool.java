import org.nd4j.linalg.api.buffer.DataType;
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
        INDArray gradient = Nd4j.create(DataType.DOUBLE, input.shape());
        int resultSize = (int) Math.ceil((float) input.size(1) / (float) kernelSize);
        INDArray result = Nd4j.create(DataType.DOUBLE, input.size(0), resultSize, resultSize);
        for(int l = 0; l < input.size(0); l++){
            for(int i = 0; i < input.size(1); i += kernelSize){
                for(int j = 0; j < input.size(2); j += kernelSize){
                    double maxVal = input.getDouble(l, j, i);
                    int maxK = 0;
                    int maxH = 0;
                    for(int k = 0; k < kernelSize && i + k < input.size(1); k++){
                        for(int h = 0; h < kernelSize && j + h < input.size(2); h++){
                            if(input.getDouble(l, j+h, i+k) > maxVal){
                                maxVal = input.getDouble(l, j+h, i+k);
                                maxK = k;
                                maxH = h;
                            }
                            gradient.putScalar(l, j + h, i + k, 0.0);
                        }
                    }
                    result.putScalar(l, j / kernelSize, i/kernelSize, maxVal);
                    gradient.putScalar(l, j+maxH, i+maxK, 1.0);
                }
            }
        }
        gradients[threadIndex] = gradient;
        return result;
    }

    @Override
    public INDArray backward(INDArray chain, int threadIndex) {
//        Matrix grad = Matrix.convertToMatrix(gradients[threadIndex]);
//        int i = 0;
//        for(int j = 0; j < gradients[threadIndex].size(0); j++){
//            for(int k = 0; k < gradients[threadIndex].size(1) * gradients[threadIndex].size(2); k++){
//                grad.matrix[j][k] = grad.matrix[j][k] * chain.getDouble(j, k % gradients[threadIndex].size(2) / kernelSize, k / gradients[threadIndex].size(2) / kernelSize);
//                i++;
//                if(i == kernelSize){
//                    i = 0;
//                }
//            }
//        }
//        gradients[threadIndex] = grad.convertToTensor();
//        return gradients[threadIndex];
        Matrix grad = Matrix.convertToMatrix(gradients[threadIndex]);
        int i = 0;
        for(int j = 0; j < gradients[threadIndex].size(0); j++){
            for(int k = 0; k < gradients[threadIndex].size(1) ; k++){
                for(int l = 0; l < gradients[threadIndex].size(2); l++) {
//                    gradients[threadIndex].putScalar(j, l, k, gradients[threadIndex].getDouble(j, l, k) * chain.getDouble(j, l / kernelSize, k / kernelSize));
                    try {
                        grad.matrix[j][grad.convert(k, l)] = grad.matrix[j][grad.convert(k, l)] * chain.getDouble(j, k / kernelSize, l / kernelSize);
                    } catch(Exception e){
                        throw e;
                    }
                    i++;
                    if (i == kernelSize) {
                        i = 0;
                    }
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