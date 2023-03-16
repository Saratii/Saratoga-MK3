package tests;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

import src.ConvolutionLayer;
import src.Matrix;

public class convTest {
    
    public static void test(){
        Matrix input = new Matrix(1, 4, 4);
        input.matrix = new Double[][]{{ 1.8811,  1.6310, -0.8117, -1.3347, 1.0977,  0.9444,  1.8712, -0.7261,-2.2977,  1.3628, -2.5223,  0.5271,-0.8172, -1.4597, -0.7961, -0.3917}};
        Matrix weights = new Matrix(1, 2, 2);
        weights.matrix = new Double[][]{{0.3008,  0.3395,-0.0640, -0.0386}};
        ConvolutionLayer conv = new ConvolutionLayer(1, 1, 2);
        conv.initialized = true;
        conv.kernals = new Matrix[1];
        conv.kernals[0] = weights;
        Matrix result = conv.forward(input);
        System.out.println(result);
        //0.6333,  0.1788,  0.3485],
        //[-0.6088, -0.5018, -0.1410],
        //[-0.7673,  0.2152,  0.0731
    }
    public static void main(String[] args){
        perf();
    }
    private static void perf(){
        Matrix veryLarge = new Matrix(1, 2000, 2000);
        veryLarge.seed();
        Matrix kernal = new Matrix(1, 10, 10);
        kernal.seed();
        long time = System.currentTimeMillis();
        Matrix somewhere = kernal.bigConvolution(veryLarge);
        long time2 = System.currentTimeMillis(); //why is the second one always smaller
        
        long time3 = System.currentTimeMillis();
        Matrix somewhere2 = kernal.bigConvolution2(veryLarge);
        long time4 = System.currentTimeMillis();
        if(!somewhere.equals(somewhere2)){
            System.out.println("Erik did a fucky wucky");
            return;
        }
        System.out.println("baseline: " + (time2 - time));
        System.out.println("is more betterer?: " + (time4 - time3));
    }
}
/*
conv(input)
tensor([[[ 0.8283, -0.1021, -0.9735],
         [ 0.5608,  0.7450,  0.2730],
         [-0.3044, -0.5066, -0.6983]]]
         
>>> conv.weight                 
Parameter containing:
tensor([[[[ 0.3008,  0.3395],
          [-0.0640, -0.0386]]]]

>>> input
tensor([[[ 1.8811,  1.6310, -0.8117, -1.3347],
         [ 1.0977,  0.9444,  1.8712, -0.7261],
         [-2.2977,  1.3628, -2.5223,  0.5271],
         [-0.8172, -1.4597, -0.7961, -0.3917]]])

>>> bias 
    tensor([-0.1845], requires_grad=True)
*/
//1.8811 * 0.3008 + 1.631 * 0.3395 + 1.0977 * -0.064 + 0.9444 * -0.0386
//1.01
