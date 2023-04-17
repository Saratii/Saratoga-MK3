package src;

import java.awt.image.BufferedImage;
import javax.swing.*;
import java.awt.*;

public class FeatureMap {
    public FeatureMap(int imageSize, BufferedImage[] images) {
        JPanel panel = new JPanel();
        GridLayout layout = new GridLayout(3, 5);
        panel.setLayout(layout);
        for(BufferedImage image : images){
            JLabel label = new JLabel(new ImageIcon(image));
            panel.add(label);
        }
        JFrame frame = new JFrame("uwu");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setSize(imageSize * 5, imageSize * 3);
        frame.setResizable(false);
        panel.setBackground(Color.green);
        frame.add(panel);
        frame.setLayout(new FlowLayout(FlowLayout.LEFT, 0, 0));
        panel.setBorder(null);
        frame.setVisible(true);
    }
}
