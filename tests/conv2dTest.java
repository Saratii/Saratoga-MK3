package tests;

import org.junit.Test;

import src.Matrix;

public class conv2dTest {
    public static double[][] conv2dAllahAkbar(Double[] input, int width, int height, Double[] kernal, int kernalWidth, int kernalHeight){
        int smallWidth = width - kernalWidth + 1;
        int smallHeight = height - kernalHeight + 1;
        double[][] output = new double[smallWidth][smallHeight];
        for(int i = 0; i < width; i++){
            for(int j = 0; j < height; j++){
                for(int kernalI = 0; kernalI < kernalWidth; kernalI ++){
                    for(int kernalJ = 0; kernalJ < kernalHeight; kernalJ ++){
                        if(i + kernalI >= smallWidth || j + kernalJ >= smallHeight){
                            continue;
                        }
                        output[i][j] += input[(i + kernalI) * height + j + kernalJ] * kernal[kernalI * kernalHeight + kernalJ];
                    }
                }
            }
        }
        return output;
    }
    @Test
    public void convOperationTest(){
        Matrix a = new Matrix(1, 10, 10);
        a.seed();
        Matrix b = new Matrix(1, 2, 2);
        b.seed();
        Matrix c = a.convolution(b);
        double[][] d = conv2dAllahAkbar(a.matrix[0], a.rows, a.cols, b.matrix[0], b.rows, b.cols);
        for(int i = 0; i < c.cols; i++){
            for(int j = 0; j < c.rows ; j++){
                assert(Math.abs(d[i][j] - c.matrix[0][i * c.cols + j]) < 0.000001);
            }
        }
    }
}

