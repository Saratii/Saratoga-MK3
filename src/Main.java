package src;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class Main {
    public static double ALPHA;
    public static void main(String[] args) throws IOException {
        BufferedImage image = ImageIO.read(new File("images/bird.png"));
        image = Matrix.makeSquare(image); //625x625 range(0:255)
        Matrix imageMatrix = Matrix.imageToMatrix(image); //(625*625 x 1) range(0:255)
        imageMatrix.normalizePixels(); //(625*625 x 1) range(-1:1)
        ConvolutionLayer conv1 = new ConvolutionLayer(10, 1, 3);
        Matrix conv1Output = conv1.forward(imageMatrix); //(623*623 x 10 x 1)  range(-3 ish:3 ish)
        Matrix maxPool1Out = conv1Output.maxPool(6);
        
        ReLU relu1 = new ReLU();
        Matrix relu1Output = relu1.forward(maxPool1Out); //104*104 x 10 x 1 range(0:3 ish)
        ConvolutionLayer conv2 = new ConvolutionLayer(10, 1, 3);
        Matrix conv2Output = conv2.forward(relu1Output);
        Matrix maxPool2Out = conv2Output.maxPool(6);
        ReLU relu2 = new ReLU();
        Matrix relu2Output = relu2.forward(maxPool2Out); //17*17 x 100 x 1 range(small ish numbers 0:5 ish)

        Matrix denseInput = Matrix.flatten(relu2Output); //17*17*100 x 1 = 28900
        Matrix expected = new Matrix(1, 5, 1);
        expected.matrix = new Double[][]{{0.0, 1.0, 0.0, 0.0, 0.0}};
        Matrix testInput = new Matrix(1, 28900, 1);
        testInput.seed();
        Model model = new Model();
        model.layers.add(new DenseLayer(denseInput.size, 256));
        model.layers.add(new ReLU());
        model.layers.add(new DenseLayer(256, 5));
        model.layers.add(new Softmax());
        ALPHA = 0.001;
        double loss = Double.POSITIVE_INFINITY;
        int i = 0;
        while(loss > 0.01){
            loss = model.forward(denseInput, expected);
            model.backward();
            System.out.println(loss);
            i++;
        }
        System.out.println("Completed in " + i + " epochs");
          //suck my dick bih
          //if you ever tell me a test failed again
          //ima smack you with my pimp cane
    }
}