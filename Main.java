import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;

public class Main {
    public static void main(String[] args) throws IOException {

        int NODES_IN_DENSE = 4;
        int NUM_FEATURE_SETS = 10;
        Matrix[] featureSets = new Matrix[NUM_FEATURE_SETS];
        BufferedImage image = ImageIO.read(new File("images/bird.png"));
        Matrix matrix = new Matrix(image.getHeight(), image.getWidth());
        image = matrix.makeSquare(image);
        matrix.imageToMatrix(image);
        BufferedImage[] featureSetImages = new BufferedImage[10];

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
        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Convolutional Layer Time: " + totalTime + "ms");

        Node[] denseLayer = new Node[NODES_IN_DENSE];
        // for(int i = 0; i < NODES_IN_DENSE; i++){
        //     denseLayer[i] = new Node(m2.matrix);
        // }

        FeatureMap map = new FeatureMap(featureSets[0].rows, featureSetImages);

    }
}