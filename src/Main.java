package src;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;

public class Main {
    public static final double ALPHA = 0.001;
    public static void main(String[] args) throws IOException {

        int NUM_FEATURE_SETS = 15;
        Matrix[] featureSets = new Matrix[NUM_FEATURE_SETS];
        BufferedImage image = ImageIO.read(new File("images/bird.png"));
        Matrix matrix = new Matrix(image.getHeight(), image.getWidth());
        image = matrix.makeSquare(image);
        matrix.imageToMatrix(image);
        BufferedImage[] featureSetImages = new BufferedImage[NUM_FEATURE_SETS];

        long startTime = System.currentTimeMillis();
        for(int i = 0 ; i < NUM_FEATURE_SETS; i++){
            Matrix kernal = new Matrix(3, 3);
            kernal.seed();
            Matrix resultant = matrix.convolution(kernal);
            resultant.ReLU();
            resultant.maxPool(2);
            BufferedImage outputImage = resultant.matrixToImage();
            featureSetImages[i] = outputImage;
            featureSets[i] = resultant;
        }
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Convolutional Layer Time: " + totalTime + "ms");

        Matrix denseInput = new Matrix(featureSets.length * featureSets[0].size, 1);
        for(int i = 0; i < featureSets.length; i++){
            for(int j = 0; j < featureSets[0].size; j++){
                denseInput.matrix[i * featureSets[0].size + j] =  featureSets[i].matrix[j];
            }
        }
        
       

        //FeatureMap map = new FeatureMap(featureSets[0].rows, featureSetImages);

        

        Matrix inputs = new Matrix(1000, 1);
        inputs.seedPositive(); 

        Matrix actual = new Matrix(5, 1);
        actual.matrix = new Double[]{0.0, 0.0, 0.0, 1.0, 0.0};

        
        DenseLayer dense = new DenseLayer(actual.rows, inputs.rows);
        Softmax soft = new Softmax();
        double lossAmount = 10;
        int i = 0;
        while(lossAmount > 0.01){
            Matrix denseOutput = dense.forward(inputs);
            System.out.println("Dense output: " + denseOutput);
            Matrix activationOutput = soft.forward(denseOutput);
            System.out.println("Softmax output: " + activationOutput);

            lossAmount = Loss.calcLoss(activationOutput, actual);
            System.out.println("Loss: " + lossAmount);
            System.out.println("Output: " + activationOutput + "\n");
            Matrix lossDerivatives = Loss.backward(activationOutput, actual);
            Matrix softDerivatives = soft.backward(lossDerivatives);
            Matrix denseDerivatives = dense.backwards(softDerivatives);
            i++;
        }
        System.out.println("In " + i + " iterations");
        

    }
}