import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class Softmax extends Layer {
    Matrix[] outputs = new Matrix[Main.numThreads];

    @Override
    public Matrix forward(Matrix inputs, int threadIndex, int batchIndexForThread) {
        INDArray input = inputs.convertToTensor();
        INDArray expArray = Transforms.exp(input.sub(input.max()));
        INDArray sumExp = expArray.sum();
        INDArray result = expArray.div(sumExp);
        outputs[threadIndex] = Matrix.convertToMatrix(result);
        return outputs[threadIndex];
    }
    //needs to be a 2x1
    // chain = 1x2
    //result = 2x1
    @Override //its softmax(i) * (1 - softmax(i))
    public Matrix backward(Matrix dvalues, int threadIndex) {
        INDArray chain = dvalues.convertToTensor();
        chain = chain.reshape(chain.size(1), chain.size(0));
        INDArray output = outputs[threadIndex].convertToTensor();
        INDArray result = Nd4j.create(DataType.DOUBLE, output.shape());
        for(int i = 0; i < output.length(); i++){
            result.putScalar(i,output.getDouble(i) * (1 - output.getDouble(i)));
        }
        return Matrix.convertToMatrix(result.transpose().mmul(chain.transpose()).transpose());
    }
    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.close();
    }
}