package src;
public class Train {
    public static void train(Matrix inputs, Matrix expected){
        DenseLayer dense1 = new DenseLayer(inputs.size, 4096);
        DenseLayer dense2 = new DenseLayer(dense1.NUM_NODES, expected.size);
        ReLU relu = new ReLU();
        Softmax soft2 = new Softmax();
        double lossAmount = 10;
        int i = 0;
        while(lossAmount > 0.01){
            Matrix dense1Output = dense1.forward(inputs);
            Matrix reluOutput = relu.forward(dense1Output);
            Matrix dense2Output = dense2.forward(reluOutput);
            Matrix activationOutput = soft2.forward(dense2Output);
            
            lossAmount = Loss.calcLoss(activationOutput, expected);
            if(i % 1 == 0){
                System.out.println("Dense output: " + dense2Output);
                System.out.println("Softmax output: " + activationOutput);
                System.out.println("Loss: " + lossAmount + "\n");
            }
            
            Matrix lossDerivatives = Loss.backward(activationOutput, expected);
            Matrix soft2Derivatives = soft2.backward(lossDerivatives);
            Matrix dense2Derivatives = dense2.backwards(soft2Derivatives);
            Matrix reluDerivatives = relu.backward(dense2Derivatives);
            dense1.backwards(reluDerivatives);
            i++;
        }
        System.out.println("In " + i + " iterations");
    }
}
