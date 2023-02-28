package src;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class Main {
    public static double ALPHA = 0.1;
    public static void main(String[] args) throws IOException {
        BufferedImage image = ImageIO.read(new File("images/bird.png"));
        image = Matrix.makeSquare(image); //625x625 range(0:255)
        Matrix imageMatrix = Matrix.imageToMatrix(image); //(625*625 x 1) range(0:255)
        imageMatrix.normalizePixels(); //(625*625 x 1) range(-1:1)
        MaxPool makeFaster = new MaxPool(70);
        imageMatrix = makeFaster.forward(imageMatrix);
        Matrix expected = new Matrix(1, 5, 1);
        expected.matrix = new Double[][]{{0.0, 1.0, 0.0, 0.0, 0.0}};
        Matrix testInput = new Matrix(1, 28900, 1);
        testInput.seed();

        Model model = new Model(); //1, 70, 70
        model.layers.add(new ConvolutionLayer(10, 1, 3)); //10, 68, 68
        model.layers.add(new ReLU()); //10, 68, 68
        model.layers.add(new MaxPool(6)); //10, 12, 12
        model.layers.add(new ConvolutionLayer(10, 1, 3)); //100, 10, 10
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(6));
        model.layers.add(new Flatten());
        model.layers.add(new DenseLayer(256));
        model.layers.add(new ReLU());
        model.layers.add(new DenseLayer(5));
        model.layers.add(new Softmax());

        double loss = Double.POSITIVE_INFINITY;
        int i = 0;
        while(loss > 0.01){
            loss = model.forward(imageMatrix, expected);
            model.backward(); //ti sounds like a vacuum cleanre

            i++;
        }
        System.out.println("Completed in " + i + " epochs");
          //ima smack you with my pimp cane
    }
}