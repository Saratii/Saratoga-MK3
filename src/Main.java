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
        ConvolutionLayer conv1 = new ConvolutionLayer(10, 1, 3);
        Matrix[] conv1Output = conv1.forward(imageMatrix); //(623*623 x 10 x 1)  range(-3 ish:3 ish)
        for(Matrix featureSet: conv1Output){
            featureSet.maxPool(6); //104*104 x 1 range(-3 ish:3 ish)
        }
        ReLU relu1 = new ReLU();
        Matrix[] relu1Output = relu1.forward(conv1Output); //104*104 x 10 x 1 range(0:3 ish)
        ConvolutionLayer conv2 = new ConvolutionLayer(10, 1, 3);

        Matrix[] conv2Output = new Matrix[conv2.NUM_FEATURE_SETS * relu1Output.length];
        for(int i = 0; i < relu1Output.length; i++){
            Matrix[] temp = conv2.forward(relu1Output[i]);
            for(int j = 0; j < temp.length; j++){
                conv2Output[i * conv2.NUM_FEATURE_SETS + j] = temp[j]; //102*102 x 10*10 x 1 range(small ish numbers -5:5 ish)
            }
        }
        for(Matrix featureSet: conv2Output){
            featureSet.maxPool(6); //17*17 x 1 range(small ish numbers -10:10 ish)
        }
        ReLU relu2 = new ReLU();
        Matrix[] relu2Output = relu2.forward(conv2Output); //17*17 x 100 x 1 range(small ish numbers 0:5 ish)

        Matrix denseInput = Matrix.flatten(relu2Output); //17*17*100 x 1 = 28900
      
        Matrix expected = new Matrix(5, 1);
        expected.matrix = new Double[]{0.0, 1.0, 0.0, 0.0, 0.0};
        // Matrix testInput = new Matrix(10, 1);
        // testInput.seed();
        Train.train(denseInput, expected);
        
    }
}