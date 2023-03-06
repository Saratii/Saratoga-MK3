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
    public Double[][] matrix;
    int z;
    int rows;
    int cols;
    public int size;
    public Matrix(int z, int rows, int cols){
        this.rows = rows;
        this.cols = cols;
        this.z = z;
        this.size = rows * cols * z;
        this.matrix = new Double[z][rows*cols];
    }
    @Override
    public String toString() {
        StringBuilder s = new StringBuilder("[");   
        for(int j = 0; j < z; j++){
            for(int i = 0; i < rows * cols; i++) {
                s.append(String.format("%.4f, ", matrix[j][i]));
            }
        }
        s.deleteCharAt(s.length()-1).deleteCharAt(s.length()-1);
        return s.append("]").toString();
    }
    public void seed(){ //fills 1d array with random values -1 to 1
        for(int j = 0; j < z; j++){
            for(int i = 0; i < rows * cols; i++){
                matrix[j][i] = r.nextDouble() * 2 - 1;
            }
        }
    }
    public void seedPositive(){
        for(int j = 0; j < z; j++){
            for(int i = 0; i < rows * cols; i++){
                matrix[j][i] = r.nextDouble();
            }
        }
    }
    public void seedZeros(){
        for(int j = 0; j < z; j++){
            for(int i = 0; i < rows * cols; i++){
                matrix[j][i] = 0.0;
            }
        }
    }
    
    public void normalizePixels(){
        for(int j = 0; j < z; j++){
            for(int i = 0; i < rows * cols; i++){
                matrix[j][i] = (matrix[j][i] - 128) / 128;
            }
        }
    }public void reverseNormalizePixels(){
        for(int j = 0; j < z; j++){
            for(int i = 0; i < rows * cols; i++){
                matrix[j][i] = matrix[j][i] * 128 + 128;
            }
        }
    }
    public void seedGaussian(){
        for(int j = 0; j < z; j++){
            for(int i = 0; i < rows * cols; i++){
                matrix[j][i] = r.nextGaussian();
            }
        }
    }
    public void seedUniform(){
        double bound = 1 / Math.sqrt(rows);
        for(int j = 0; j < z; j++){
            for(int i = 0; i < rows * cols; i++){
                matrix[j][i] = (r.nextDouble() - 0.5) * 2 * bound;
            }
        }
    }
    public static Matrix imageToMatrix(BufferedImage image) throws IOException{
        int width = image.getWidth();
        int height = image.getHeight();
        Matrix result = new Matrix(1, height, width);
        for(int x = 0; x < width; x++){
            for(int y = 0; y < height; y++){
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xff;
                int g = (rgb >> 8) & 0xff;
                int b = rgb & 0xff;
                double gray = 0.2989 * r + 0.5870 * g + 0.1140 * b;
                result.matrix[0][y* width + x] = gray;
            }
        }
        return result;
    }
    public static BufferedImage makeSquare(BufferedImage image){
        if(image.getHeight() > image.getWidth()){
            BufferedImage im = image.getSubimage(0, image.getHeight()/2 - image.getWidth()/2, image.getWidth(), image.getWidth());
            return im;
        } else{
            BufferedImage im = image.getSubimage(image.getWidth()/2 - image.getHeight()/2, 0, image.getHeight(), image.getHeight());
            return im;
        }
        
    }
    public BufferedImage matrixToImage(){
        BufferedImage image = new BufferedImage(rows, cols, BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < cols; x++){
            for(int y = 0; y < rows; y++){
                Double value = matrix[0][y*cols+x];
                Color grey = new Color(value.intValue(), value.intValue(), value.intValue());
                int rgb = grey.getRGB();
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }
    public static double maxValue(Double[] inputs){
        double max = Double.NEGATIVE_INFINITY;
        for(double value : inputs){
            if(value > max){
                max = value;
            }
        }
        return max;
    }
    public static double minValue(Double[] inputs){
        double min = Double.POSITIVE_INFINITY;
        for(double value : inputs){
            if(value < min){
                min = value;
            }
        }
        return min;
    }
    public static double sum(Double[] inputs){
        double sum = 0;
        for(int i = 0; i < inputs.length; i++){
            sum += inputs[i];
        }
        return sum;
    }
    
    public Matrix convolution(Matrix kernal){
        Matrix resultant = new Matrix(z * kernal.z, rows - kernal.rows + 1, cols - kernal.cols + 1);
        int thisJ, thisK, iz, imz, i, j, k, jj, kk;
        for(i = 0; i < resultant.z; i++){
            iz = i/z;
            imz = i % z;
            for(j = 0; j < resultant.rows; j++){
                for(k = 0; k < resultant.cols; k++){
                    double dotProd = 0;
                    for(jj = 0; jj < kernal.rows; jj++){
                        thisJ = j + jj;
                        int rowIndex = thisJ * cols;
                        int kernalRowIndex = jj * kernal.cols;
                        for(kk = 0; kk < kernal.cols; kk++){
                            thisK = k + kk;
                            dotProd += matrix[imz][rowIndex + thisK] * kernal.matrix[iz][kernalRowIndex + kk];
                        }
                    }
                    resultant.matrix[i][resultant.convert(j, k)] = dotProd;
                }
            }
        }
        return resultant; 
    }
    
    public Matrix bigConvolution(Matrix kernal){
        Matrix resultant = new Matrix(z * kernal.z, rows + kernal.rows - 1, cols + kernal.cols - 1);
        int thisJ, thisK, iz, imz, i, j, k, jj, kk;
        for(i = 0 ; i < resultant.z; i++){
            iz = i/z;
            imz = i % z;
            for(j = 0; j < resultant.rows; j++){
                int jjStart = Math.max(0, kernal.rows - j - 1);
                int jjEnd = Math.min(kernal.rows, rows - j + kernal.rows - 1);
                int resultantRowIndex = j * resultant.cols;
                for(k = 0; k < resultant.cols; k++){
                    int kkStart = Math.max(0, kernal.cols - k - 1);
                    int kkEnd = Math.min(kernal.cols, cols - k + kernal.cols - 1);
                    double dotProd = 0;
                    for(jj = jjStart; jj < jjEnd; jj++){
                        thisJ = j + jj - kernal.rows + 1;
                        int rowIndex = thisJ * cols;
                        int kernalRowIndex = jj * kernal.cols;
                        for(kk = kkStart; kk < kkEnd; kk++){
                            thisK = k + kk - kernal.cols + 1;
                            dotProd += matrix[imz][rowIndex + thisK] * kernal.matrix[iz][kernalRowIndex + kk];
                        }
                    }
                    resultant.matrix[i][resultantRowIndex + k] = dotProd;
                }
            }
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
            sum += matrix[0][i] * matrixB.matrix[0][i];
        }
        return sum;
    }
    public void ReLU(){
        for(int i = 0; i < size; i++){
            matrix[0][i] = (matrix[0][i] < 0) ? 0.0: matrix[0][i];
        }
    }
    
    public int convert(int x, int y){
        return y * cols + x;
    }

    public Matrix multiply(Matrix matrixB) {
        Matrix result = new Matrix(1, rows, matrixB.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < matrixB.cols; j++) {
                result.matrix[0][i*matrixB.cols + j] = 0.0;
                for (int k = 0; k < cols; k++) {
                    result.matrix[0][i * matrixB.cols + j] += matrix[0][i * cols + k] * matrixB.matrix[0][k * matrixB.cols + j];
                }
            }
        }
        return result;
    }
    public void reverse(){
        for(int i = 0; i < z; i++){
            int left = 0;
            int right = matrix[i].length - 1;
            double temp = matrix[i][left];
            matrix[i][left] = matrix[i][right];
            matrix[i][right] = temp;
            left++;
            right--;
        }
    }
    public String getSize(){
        return "[" + z + ", " + rows + ", " + cols + "] " + size;
    }
    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = resizedImage.createGraphics();
        graphics2D.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        graphics2D.dispose();
        return resizedImage;
    }
}