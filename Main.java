import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedImage image = ImageIO.read(new File("images\\dog1.png"));
        Matrix matrix = new Matrix(image.getHeight(), image.getWidth());
        Matrix kernal = new Matrix(3, 3);
        kernal.seedScharr();
        image = matrix.squareImage(image);
        matrix.imageToMatrix(image);
        matrix.convolution(kernal);
        matrix.ReLU();
        image = matrix.matrixToImage();
        matrix.display(image);
    }
}