package src;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;

public class Image {
    Matrix imageData;
    Matrix label;
    String stringLabel;
    BufferedImage bufferedImage;
    public Image(File file, String label) throws IOException{
        this.bufferedImage = ImageIO.read(file);
        this.imageData = Matrix.imageToMatrix(bufferedImage);
        this.label = new Matrix(1, 2, 1);
        this.stringLabel = label;
        if(label == "dolphin"){
            this.label.matrix = new Double[][]{{1.0, 0.0}};
        } else if(label == "antelope"){
            this.label.matrix = new Double[][]{{0.0, 1.0}};
        }
    }
    public static Matrix[] shuffle(Matrix[] arr){
        Random rnd = new Random();
        for (int i = arr.length - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            Matrix temp = arr[index];
            arr[index] = arr[i];
            arr[i] = temp;
        }
        return arr;
    }
}
