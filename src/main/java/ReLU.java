import org.nd4j.linalg.api.ndarray.INDArray;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class ReLU extends Layer {
    INDArray[] values = new INDArray[Main.numThreads];

    @Override
    public INDArray forward(INDArray input, int threadIndex, int batchIndexForThread) {
        for(int i = 0; i < input.length(); i++){
            input.putScalar(i, Math.max(input.getDouble(i), 0));
        }
        values[threadIndex] = input;
        return values[threadIndex];
    }

    @Override
    public INDArray backward(INDArray chain, int threadIndex) {
        INDArray inputs = values[threadIndex];
        for(int i = 0; i < chain.length(); i++){
            chain.putScalar(i, inputs.getDouble(i) > 0 ? chain.getDouble(i) : 0.0);
        }
        return chain;
    }

    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("src/main/logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.close();
    }
}