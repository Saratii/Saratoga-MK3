package src;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Random;

public class Dropout extends Layer {
    Matrix[] dropOuts;
    final double percentageLost;
    final double scale;
    Random r = new Random();

    public Dropout(double percentageLost) {
        dropOuts = new Matrix[Main.numThreads];
        this.percentageLost = percentageLost;
        scale = 1 / (1 - percentageLost);
    }

    @Override
    public Matrix forward(Matrix values, int threadIndex) {
        if(isClassifying){
            return values;
        }
        Matrix result = new Matrix(values.z, values.rows, values.cols);
        dropOuts[threadIndex] = new Matrix(values.z, values.rows, values.cols);
        for(int j = 0; j < values.z; j++){
            for(int i = 0; i < values.innerSize; i++){
                dropOuts[threadIndex].matrix[j][i] = r.nextDouble() > percentageLost ? scale : 0.0;
                result.matrix[j][i] = values.matrix[j][i] * dropOuts[threadIndex].matrix[j][i];
            }
        }
        return result;
    }

    @Override
    public Matrix backward(Matrix dvalues, int threadIndex) {
        Matrix result = new Matrix(dvalues.z, dvalues.rows, dvalues.cols);
        for(int j = 0; j < dvalues.z; j++){
            for(int i = 0; i < dvalues.innerSize; i++){
                if(dropOuts[threadIndex].matrix[j][i] > 0){
                    result.matrix[j][i] = dvalues.matrix[j][i] / dropOuts[threadIndex].matrix[j][i];
                } else {
                    result.matrix[j][i] = 0.0;
                }
            }
        }
        return result;
    }

    @Override
    public void write() throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("logs/log-" + this, "UTF-8");
        writer.println(this);
        writer.println("Dropout Percentage{" + percentageLost + "}");
        writer.close();
    }
}