import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedImage image = ImageIO.read(new File("images/bird.png"));
        // Matrix matrix = new Matrix(image.getHeight(), image.getWidth());
        Matrix matrix = new Matrix(5, 5);
        matrix.testSeed();
        Matrix kernal = new Matrix(3, 3);
        kernal.seedScharr();
        image = matrix.squareImage(image);
        matrix.imageToMatrix(image);
        matrix.convolution(kernal);
        matrix.ReLU();
        matrix.maxPool(2);
        image = matrix.matrixToImage();
        matrix.display(image);

        // BufferedImage i2 = ImageIO.read(new File("images/bird.png"));
        // Matrix m2 = new Matrix(i2.getHeight(), i2.getWidth());
        // Matrix k2 = new Matrix(3, 3);
        // k2.seedHorizontal();
        // i2 = m2.squareImage(i2);
        // m2.imageToMatrix(i2);
        // m2.convolution(k2);
        // m2.ReLU();
        // i2 = m2.matrixToImage();
        // m2.display(i2);

        // BufferedImage i3 = ImageIO.read(new File("images/bird.png"));
        // Matrix m3 = new Matrix(i3.getHeight(), i3.getWidth());
        // Matrix k3 = new Matrix(3, 3);
        // k3.seedVertical();
        // i3 = m3.squareImage(i3);
        // m3.imageToMatrix(i3);
        // m3.convolution(k3);
        // m3.ReLU();
        // i3 = m3.matrixToImage();
        // m3.display(i3);
        

    }
}