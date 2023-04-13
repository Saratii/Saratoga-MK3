package src;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import java.awt.FlowLayout;

public class Image {
    Matrix imageData;
    Matrix label;
    String stringLabel;
    BufferedImage bufferedImage;
    public Image(Path path, String label) throws IOException{
        this.bufferedImage = Matrix.resizeImage(Matrix.makeSquare(ImageIO.read(path.toFile())), 40, 40);
        this.imageData = Matrix.imageToMatrix(bufferedImage);
        imageData.normalizePixels();
        this.label = new Matrix(1, 2, 1);
        this.stringLabel = label;
        if(label == "dog"){
            this.label.matrix = new Double[][]{{1.0, 0.0}};
        } else if(label == "elefante"){
            this.label.matrix = new Double[][]{{0.0, 1.0}};
        }
    }
    public static void shuffle(List<Image> arr){
        Random r = new Random(0);
        for (int i = arr.size() - 1; i > 0; i--) {
            int index = r.nextInt(i + 1);
            Image temp = arr.get(index);
            arr.set(index, arr.get(i));
            arr.set(i, temp);
        }
    }
    public void show(){
        JFrame frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        frame.getContentPane().add(new JLabel(new ImageIcon(bufferedImage)));
        frame.pack();
        frame.setVisible(true);
    }
}
