import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedImage image = ImageIO.read(new File("images/bird.png"));
        Matrix matrix = new Matrix(image.getHeight(), image.getWidth());
    
        Matrix kernal = new Matrix(3, 3);
        kernal.seedScharr();
        image = matrix.makeSquare(image);
        matrix.imageToMatrix(image);
        matrix.convolution(kernal);
        matrix.ReLU();
        // matrix.maxPool(2);
        image = matrix.matrixToImage();
        matrix.display(image);

        BufferedImage i2 = ImageIO.read(new File("images/bird.png"));
        Matrix m2 = new Matrix(i2.getHeight(), i2.getWidth());
        Matrix k2 = new Matrix(3, 3);
        k2.seedScharr();
        i2 = m2.makeSquare(i2);
        m2.imageToMatrix(i2);
        m2.convolution(k2);
        m2.ReLU();
        m2.maxPool(3);
        i2 = m2.matrixToImage();
        m2.display(i2);


    }
}