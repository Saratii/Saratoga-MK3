package src;

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
    private Random r = new Random();
    public Double[] matrix;
    int rows;
    int cols;
    public int size;
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
    public void seedPositive(){
        for(int i = 0; i < size; i++){
            matrix[i] = r.nextDouble();
        }
    }
    public void testSeed(){ //fills 1d array with random values 0 - 1
        for(int i = 0; i < size; i++){
            matrix[i] = (double) i/10;
        }
    }
 
    public Matrix imageToMatrix(BufferedImage image) throws IOException{
        int width = image.getWidth();
        int height = image.getHeight();
        Matrix result = new Matrix(height, width);
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
                result.matrix[convert(x, y)] = ((double) (r + g + b) / 3) / 127.5 + 1;
            }
        }
        return result;
    }
    public BufferedImage makeSquare(BufferedImage image){
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
                Double value = matrix[y*cols+x];
                value = (value + 1) * 127.5;
                if(value > 255){
                    value = 255.0;
                }
                Color grey = new Color(value.intValue(), value.intValue(), value.intValue());
                int rgb = grey.getRGB();
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }
    public Matrix convolution(Matrix kernal){
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
        return resultant; 
    }
    public void display(BufferedImage image){
        // if(frame==null){
            frame=new JFrame();
            frame.setTitle("UwU");
            frame.setSize(image.getWidth(), image.getHeight());
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            frame.add(new JLabel(new ImageIcon(image)));
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
            matrix[i] = (matrix[i] < 0) ? 0.0: matrix[i];
        }
    }
    public void maxPool(int kernalSize){
        int resultSize = (int) Math.ceil((float) rows / (float) kernalSize);
        Matrix result = new Matrix(resultSize, resultSize);

        for(int i = 0; i < rows; i += kernalSize){
            for(int j = 0; j < cols; j += kernalSize){
                double maxVal = matrix[convert(i, j)];
                for(int k = 0; k < kernalSize && i + k < rows; k++){
                    for(int h = 0; h < kernalSize && j + h < cols; h++){
                        if(matrix[convert(i + k, j + h)] > maxVal){
                            maxVal = matrix[convert(i + k, j + h)];
                        }
                    } 
                }
                result.matrix[result.convert(i / kernalSize, j / kernalSize)] = maxVal;
            }
        }
        size = result.size;
        cols = result.cols;
        rows = result.rows;
        matrix = result.matrix;
    }
    public int convert(int x, int y){
        return y * cols + x;
    }

    public Matrix multiply(Matrix matrixB) {
        Matrix result = new Matrix(rows, matrixB.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < matrixB.cols; j++) {
                result.matrix[i*matrixB.cols + j] = 0.0;
                for (int k = 0; k < cols; k++) {
                    result.matrix[i * matrixB.cols + j] += matrix[i * cols + k] * matrixB.matrix[k * matrixB.cols + j];
                }
            }
        }
        return result;
    }
}