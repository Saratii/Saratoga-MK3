import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class Softmax extends Layer {
    INDArray[] outputs = new INDArray[Main.numThreads];

    @Override
    public INDArray forward(INDArray input, int threadIndex, int batchIndexForThread) {
        INDArray expArray = Transforms.exp(input.sub(input.max()));
        INDArray sumExp = expArray.sum();
        INDArray result = expArray.div(sumExp);
        outputs[threadIndex] = result;
        return outputs[threadIndex];
    }
    @Override
    public Matrix backward(Matrix dvalues, int threadIndex) {
        INDArray chain = dvalues.convertToTensor();
        INDArray result = Nd4j.create(DataType.DOUBLE, chain.length(), chain.length());
        for(int i = 0; i < chain.length(); i++){
            for(int j = 0; j < chain.length(); j++){
                result.putScalar(j, i, outputs[threadIndex].getDouble(i, 0) * ((i == j ? 1 : 0) - outputs[threadIndex].getDouble(j, 0)));
            }
        }
        return Matrix.convertToMatrix(result.mmul(chain));
    }
    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.close();
    }
}