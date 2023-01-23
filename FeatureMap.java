import java.awt.image.BufferedImage;
import javax.swing.*;
import java.awt.*;

public class FeatureMap {
    public FeatureMap(int imageSize, BufferedImage[] images){
        JFrame frame = new JFrame("UwU");
        JPanel panel = new JPanel();
        frame.getContentPane();
        // panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        panel.setBackground(Color.green);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.add(panel);
        frame.setSize(imageSize * 6, imageSize * 4);
        frame.setVisible(true);
        for(int j = 0; j < 3; j++){
            for(int i = 0; i < 5; i++){
                // JLabel label = new JLabel(new ImageIcon(images[j * 5 + i]));
                JLabel label = new JLabel(new ImageIcon(images[0]));
                label.setBounds(imageSize * i, j * imageSize, imageSize, imageSize);
                panel.add(label);
            }
        }
        
    }

}
