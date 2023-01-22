
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Random;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.*;
import java.awt.*;

public class Matrix {
    private static JFrame frame;
    private static JLabel label;
    private Random r = new Random();
    private Double[] matrix;
    int rows;
    int cols;
    int size;
    public Matrix(int rows, int cols){
        this.rows = rows;
        this.cols = cols;
        this.size = rows * cols;
        this.matrix = new Double[size];
    }
    @Override
    public String toString() { // returns a string version of the array for viewing
        String s = "[";
        for(int i = 0; i < size - 1; i++){
            s = s + (matrix[i] + ", ");
        }
        return s + matrix[size - 1] + "]";
    }
    public void seed(){ //fills 1d array with random values 0 - 1
        
        for(int i = 0; i < size; i++){
            matrix[i] = r.nextDouble() * 2 - 1;
        }
    }
    public void testSeed(){ //fills 1d array with random values 0 - 1
        for(int i = 0; i < size; i++){
            matrix[i] = (double) i/10;
        }
    }
    public void seedVertical(){ //fills 1d array with random values 0 - 1
        for(int i = 0; i < size; i++){
            matrix[i] = (double) ((i % cols == 0) ? -1 : (i % cols == cols - 1) ? 1 : 0);
        }
    }
    public void seedHorizontal(){ //fills 1d array with random values 0 - 1
        matrix[0] = 1.0;
        matrix[1] = 1.0;
        matrix[2] = 1.0;
        matrix[3] = 0.0;
        matrix[4] = 0.0;
        matrix[5] = 0.0;
        matrix[6] = -1.0;
        matrix[7] = -1.0;
        matrix[8] = -1.0;
    }
    public void seedSobel(){ //fills 1d array with random values 0 - 1
        matrix[0] = 1.0;
        matrix[1] = 0.0;
        matrix[2] = -1.0;
        matrix[3] = 2.0;
        matrix[4] = 0.0;
        matrix[5] = -2.0;
        matrix[6] = 1.0;
        matrix[7] = 0.0;
        matrix[8] = -1.0;
    }
    public void seedScharr(){ //fills 1d array with random values 0 - 1
        matrix[0] = 3.0;
        matrix[1] = 0.0;
        matrix[2] = -3.0;
        matrix[3] = 5.0;
        matrix[4] = 0.0;
        matrix[5] = -5.0;
        matrix[6] = 3.0;
        matrix[7] = 0.0;
        matrix[8] = -3.0;
    }
    public void seedDiagnonal(){ //fills 1d array with random values 0 - 1
        matrix[0] = 0.0;
        matrix[1] = 1.0;
        matrix[2] = 0.0;
        matrix[3] = -1.0;
        matrix[4] = 0.0;
        matrix[5] = 1.0;
        matrix[6] = 0.0;
        matrix[7] = -1.0;
        matrix[8] = 0.0;
    }
    public void seedSharpen(){ //fills 1d array with random values 0 - 1
        if(size != 25){
            System.out.println("wrong kernal size");
        }
        for(int i = 0; i < size; i++){
            matrix[i] = 0.0;
        }
        matrix[7] = -1.0;
        matrix[11] = -1.0;
        matrix[12] = 5.0;
        matrix[13] = -1.0;
        matrix[17] = -1.0; 
    }
    public void seedZeros(){
        for(int i = 0; i < size; i++){
            matrix[i] = 0.0;
        }
    }
    public void seedGrey(){
        if(size != 3){
            System.out.println("wrong kernal size");
            for(int i = 0; i < size; i++){
                matrix[i] = 0.0;
            }
            matrix[0] = 1.0;
            matrix[2] = -1.0;
            matrix[3] = 2.0;
            matrix[5] = -2.0;
            matrix[6] = 1.0;
            matrix[8] = -1.0;
        }
    }
    public void imageToMatrix(BufferedImage image) throws IOException{
        int width = image.getWidth();
        int height = image.getHeight();
        for(int x = 0; x < width; x++){
            for(int y = 0; y < height; y++){
                int p = image.getRGB(x, y);
                int a = (p>>24)&0xff;
                int r = (p>>16)&0xff;
                int g = (p>>8)&0xff;
                int b = p&0xff;
                int avg = (r + g + b) / 3;
                p = (a<<24) | (avg<<16) | (avg<<8) | avg;
                image.setRGB(x, y, p);
                matrix[convert(x, y)] = (double) (r + g + b) / 3;
            }
        }
    }
    public BufferedImage squareImage(BufferedImage image){
        if(rows > cols){
            BufferedImage im = image.getSubimage(0, rows/2 - cols/2, cols, cols);
            rows = cols;
            return im;
        } else{
            BufferedImage im = image.getSubimage(cols/2 - rows/2, 0, rows, rows);
            cols = rows;
            return im;
        }
        
    }
    public BufferedImage matrixToImage(){
        BufferedImage image = new BufferedImage(rows, cols, BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < cols; x++){
            for(int y = 0; y < rows; y++){
                Color grey = new Color(matrix[y * cols + x].intValue(), matrix[y * cols + x].intValue(), matrix[y * cols + x].intValue());
                int rgb = grey.getRGB();
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }
    public void convolution(Matrix kernal){
        Matrix resultant = new Matrix(rows - kernal.rows + 1, cols - kernal.cols + 1);
        for(int i = 0; i < resultant.size; i++){
            Matrix subby = new Matrix(kernal.rows, kernal.cols);
            for( int j = 0; j < subby.rows; j++){
                for( int k = 0; k < subby.cols; k++){
                    subby.matrix[j * subby.cols + k] = matrix[i / resultant.cols * cols + i % resultant.cols + j * cols + k];
                }
            }
            resultant.matrix[i] = subby.dotProduct(kernal);
        }
        matrix = resultant.matrix;
        size = resultant.size;
        cols = resultant.cols;
        rows = resultant.rows;
    }
    public void display(BufferedImage image){
        // if(frame==null){
            frame=new JFrame();
            frame.setTitle("UwU");
            frame.setSize(image.getWidth(), image.getHeight());
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            label=new JLabel();
            label.setIcon(new ImageIcon(image));
            frame.getContentPane().add(label, BorderLayout.CENTER);
            frame.setLocationRelativeTo(null);
            frame.pack();
            frame.setVisible(true);
        // } else label.setIcon(new ImageIcon(image));
    }
    public double dotProduct(Matrix matrixB){
        double sum = 0;
        for(int i = 0; i < size; i++){
            sum += matrix[i] * matrixB.matrix[i];
        }
        return sum;
    }
    public void ReLU(){
        for(int i = 0; i < size; i++){
            matrix[i] = (matrix[i] < 255) ? ((matrix[i] > 0) ? matrix[i] : 0) : 255;
            // matrix[i] = (matrix[i] / (255 / 105.0)) + 150;
        }
    }
    public void maxPool(int kernalSize){
        int resultSize = (int) Math.ceil((float) rows / (float) kernalSize);
        Matrix result = new Matrix(resultSize, resultSize);

    
    }
    public int convert(int x, int y){
        return y * cols + x;
    }
}