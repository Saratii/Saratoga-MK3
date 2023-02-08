package src;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;

public class Main {
    public static final double ALPHA = 0.001;
    public static void main(String[] args) throws IOException {
        BufferedImage image = ImageIO.read(new File("images/bird.png"));
        image = Matrix.makeSquare(image); //625x625 range(0:255)
        Matrix imageMatrix = Matrix.imageToMatrix(image); //(625*625 x 1) range(0:255)
        imageMatrix.normalizePixels(); //(625*625 x 1) range(-1:1)
        ConvolutionLayer conv = new ConvolutionLayer(imageMatrix, 10, 1, 3);
        Matrix[] convOutput = conv.forward(); //(623*623 x 10 x 1)  range(-3 ish:3 ish)
        for(Matrix featureSet: convOutput){
            featureSet.maxPool(3); //208x208 range(-3 ish:3 ish)
        }
        Matrix data = Matrix.flatten(convOutput); //208*208*10 x 1 range(-3 ish:3 ish)
        
        
        
        // convOutput.maxPool(10);
        // convOutput.ReLU();
        // Matrix inputs = new Matrix(400000, 1);
        // inputs.seedPositive(); 

        // Matrix expected = new Matrix(5, 1);
        // expected.matrix = new Double[]{0.0, 0.0, 0.0, 1.0, 0.0};

        // Train.train(inputs, expected);
    }
}