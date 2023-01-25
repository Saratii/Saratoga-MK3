import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;

public class Main {
    public static void main(String[] args) throws IOException {

        int NODES_IN_DENSE = 4;
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

        startTime = System.currentTimeMillis();
        Matrix denseInput = new Matrix(featureSets.length * featureSets[0].size, 1);
        for(int i = 0; i < featureSets.length; i++){
            for(int j = 0; j < featureSets[0].size; j++){
                denseInput.matrix[i * featureSets[0].size + j] =  featureSets[i].matrix[j];
            }
            
        }
        System.out.println(denseInput.rows);
     
        // for(int i = 0; i < NODES_IN_DENSE; i++){
        //     denseLayer[i] = new Node(m2.matrix);
        // }

        FeatureMap map = new FeatureMap(featureSets[0].rows, featureSetImages);

    }
}