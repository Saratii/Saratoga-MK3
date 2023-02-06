package src;
public class Train {
    public static void train(Matrix inputs, Matrix expected){
        DenseLayer dense = new DenseLayer(expected.rows, inputs.rows);
        Softmax soft = new Softmax();
        double lossAmount = 10;
        int i = 0;
        while(lossAmount > 0.01){
            Matrix denseOutput = dense.forward(inputs);
            System.out.println("Dense output: " + denseOutput);
            Matrix activationOutput = soft.forward(denseOutput);
            System.out.println("Softmax output: " + activationOutput);

            lossAmount = Loss.calcLoss(activationOutput, expected);
            System.out.println("Loss: " + lossAmount);
            System.out.println("Output: " + activationOutput + "\n");
            Matrix lossDerivatives = Loss.backward(activationOutput, expected);
            Matrix softDerivatives = soft.backward(lossDerivatives);
            Matrix _denseDerivatives = dense.backwards(softDerivatives);
            i++;
        }
        System.out.println("In " + i + " iterations");
    }

}
