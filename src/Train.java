package src;
import java.awt.image.BufferedImage;
public class Train {
    public static void train(Matrix inputs, Matrix expected){
        DenseLayer dense1 = new DenseLayer(8, inputs.size);
        DenseLayer dense2 = new DenseLayer(expected.size, dense1.NUM_NODES);
        ReLU relu = new ReLU();
        Softmax soft2 = new Softmax();
        double lossAmount = 10;
        int i = 0;
        while(lossAmount > 0.01){
            Matrix dense1Output = dense1.forward(inputs);
            Matrix reluOutput = relu.forward(dense1Output);
            Matrix dense2Output = dense2.forward(reluOutput);
            System.out.println("Dense output: " + dense2Output);
            Matrix activationOutput = soft2.forward(dense2Output);
            System.out.println("Softmax output: " + activationOutput);

            lossAmount = Loss.calcLoss(activationOutput, expected);
            System.out.println("Loss: " + lossAmount);
            System.out.println("Output: " + activationOutput + "\n");
            Matrix lossDerivatives = Loss.backward(activationOutput, expected);
            Matrix soft2Derivatives = soft2.backward(lossDerivatives);
            Matrix dense2Derivatives = dense2.backwards(soft2Derivatives);
            Matrix reluDerivatives = relu.backward(dense2Derivatives);
            Matrix dense1Derivatives = dense1.backwards(reluDerivatives);
            i++;
        }
        System.out.println("In " + i + " iterations");
    }
}
