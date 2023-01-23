import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Random;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.*;
import java.awt.*;

public class FeatureMap {
    private static JFrame frame;
    public FeatureMap(int size, BufferedImage[] images){
        frame=new JFrame();
        frame.setTitle("UwU");
        frame.setSize(size * 2, size * 2);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        for(int i = 0; i < 5; i++){
            JLabel label = new JLabel(new ImageIcon(images[i]));
            label.setLocation(i * size * 0, 0);
            frame.add(label);
            
        }
        frame.setVisible(true);

    }
}
