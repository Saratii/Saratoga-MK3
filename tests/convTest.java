package tests;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

import src.ConvolutionLayer;
import src.Matrix;

public class convTest {
    
    public static void test(){
        Matrix input = new Matrix(1, 4, 4);
        input.matrix = new Double[][]{{-0.3150, -1.2068,  1.2287, -0.9581 ,0.9740,  1.7860,  1.7604, -0.3961, -0.9415,  1.1161, -0.6893,  0.3900, -1.1575, -0.1022,  1.2918,  0.6248}};
        Matrix weights = new Matrix(1, 2, 2);
        weights.matrix = new Double[][]{{0.0220, -0.2466, 0.2102,  0.1938}};
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
        test();
    }
}
/*import torch

>>> conv = nn.Conv2d(1, 1, 2, stride=1)
>>> input = torch.randn(1, 4, 4)
print(conv(input))
tensor([[[ 0.6333,  0.1788,  0.3485],
         [-0.6088, -0.5018, -0.1410],
         [-0.7673,  0.2152,  0.0731]]], grad_fn=<SqueezeBackward1>)

>>> input
tensor([[[-0.3150, -1.2068,  1.2287, -0.9581],
         [ 0.9740,  1.7860,  1.7604, -0.3961],
         [-0.9415,  1.1161, -0.6893,  0.3900],
         [-1.1575, -0.1022,  1.2918,  0.6248]]])

>>> conv.weight
Parameter containing:
tensor([[[[ 0.0220, -0.2466],
          [ 0.2102,  0.1938]]]], requires_grad=True)

*/
