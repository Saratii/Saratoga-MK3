package tests;

import org.junit.Test;

import src.ConvolutionLayer;
import src.Flatten;
import src.Matrix;

public class conv2dTest {
    public static double[][] conv2d(Double[] input, int width, int height, Double[] kernal, int kernalWidth, int kernalHeight){
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
    public void convTest() {
        ConvolutionLayer conv = new ConvolutionLayer(3, 2, 1, 2);
        Matrix input = new Matrix(3, 10, 10);
        input.matrix = new Double[][]{
        {1.4437e+00,  2.6605e-01,  1.6646e-01,  8.7438e-01, -1.4347e-01,
        -1.1161e-01, -6.1358e-01,  3.1593e-02,  2.0050e+00,  5.3737e-02,
        6.1806e-01, -4.1280e-01, -8.4106e-01, -2.3160e+00, -1.0231e-01,
        7.9244e-01,  5.6272e-01,  2.5963e-01, -1.7396e-01, -6.7875e-01,
        9.3826e-01,  4.8887e-01, -6.7309e-01,  8.7283e-01, -1.2001e+00,
        -4.7859e-03, -5.1807e-01, -3.0670e-01, -1.5810e+00,  1.7066e+00,
        -4.4622e-01,  7.4402e-01, -5.7308e-01, -5.5536e-01,  5.9432e-01,
        1.5419e+00,  5.0733e-01, -5.9103e-01, -5.6925e-01,  9.1997e-01,
        -6.9073e-02, -4.9493e-01, -1.4959e+00, -1.9384e-01,  4.4551e-01,
        1.3253e+00, -1.6293e+00, -5.4974e-01,  1.7067e+00,  2.3804e+00,
        -1.1256e+00, -3.1700e-01, -1.0925e+00, -8.5194e-02, -9.3348e-02,
        6.8705e-01, -1.5991e+00,  1.8487e-02, -7.5043e-01,  1.8541e-01,
        6.2114e-01,  6.3818e-01, -2.4600e-01,  2.3025e+00,  1.1687e+00,
        3.9450e-01,  1.9415e+00,  7.9150e-01, -2.0252e-02, -4.3717e-01,
        1.6459e+00, -1.3602e+00,  9.6630e-01,  1.6248e+00, -3.6562e-01,
        -1.3024e+00,  9.9403e-02,  4.4182e-01,  2.4693e-01,  7.6887e-02,
        2.8227e-01,  4.3423e-01, -8.0249e-01, -1.2952e+00, -7.5018e-01,
        -1.3120e+00, -2.1883e-01, -2.4351e+00, -4.2883e-01,  2.3292e-01,
        7.9689e-01, -1.8484e-01, -3.7015e-01, -1.2103e+00, -6.2270e-01,
        -4.6372e-01, -6.0367e-01, -1.2788e+00,  9.2950e-02, -6.6610e-01},

        {6.0805e-01, -7.3002e-01, -8.8338e-01, -4.1891e-01,  4.7656e-01,
        -1.0163e+00,  1.8037e-01,  1.0833e-01, -7.5482e-01,  2.4432e-01,
        -7.7326e-02,  1.1640e-01,  7.2980e-01, -1.8453e+00, -2.5020e-02,
        1.3694e+00,  2.6570e+00,  9.8512e-01, -2.5964e-01,  1.1834e-01,
        -1.1428e+00,  3.7586e-02,  2.6963e+00,  1.2358e+00,  5.4283e-01,
        5.2553e-01,  1.9220e-01, -7.7216e-01,  1.6268e+00,  1.7227e-01,
        -1.6115e+00, -4.7945e-01, -1.4335e-01, -3.1729e-01,  9.6715e-01,
        -9.9108e-01,  5.4361e-01,  7.8804e-02,  8.6286e-01, -1.9490e-02,
        9.9105e-01, -7.7773e-01,  3.1405e-01,  2.1333e-01,  1.9159e+00,
        6.9020e-01, -2.3217e+00, -1.1964e+00,  1.9703e-01, -1.1773e+00,
        -6.6145e-02, -3.5836e-01, -1.3952e+00,  4.7512e-01, -8.1373e-01,
        9.2424e-01, -2.4734e-01, -1.4154e+00, -1.0787e+00, -7.2091e-01,
        5.8669e-01,  1.5830e-01,  1.1025e-01, -8.1881e-01,  6.3277e-01,
        -1.9169e+00, -5.5963e-01,  5.3347e-01,  7.8173e-01,  9.8969e-01,
        4.1471e-01, -1.5090e+00,  2.0360e+00,  1.3159e-01, -7.4395e-02,
        -1.0922e+00, -5.1006e-01, -4.7489e-01, -6.3340e-01, -1.4677e+00,
        -8.7848e-01, -2.0784e+00,  1.1711e+00,  9.7496e-02,  1.1931e-02,
        3.3977e-01, -2.6345e-01,  1.2805e+00,  1.9395e-02, -8.8080e-01,
        9.5522e-01,  1.2836e+00,  1.3384e+00, -2.7940e-01, -5.5183e-01,
        -2.8891e+00, -1.5100e+00,  1.0241e+00, -3.4334e-01,  1.5713e+00},

        {1.7001e+00,  3.4622e-01,  9.7112e-01,  1.4503e+00, -5.1909e-02,
        -6.2843e-01, -1.4201e-01, -5.3415e-01, -9.6095e-01, -6.3750e-01,
        7.4724e-02,  5.5997e-01,  5.3140e-01,  1.2351e+00, -1.4777e+00,
        -1.7557e+00,  1.5346e+00, -3.2062e-03, -1.6034e+00,  5.8098e-02,
        -6.3025e-01,  7.4664e-01, -6.4025e-02,  1.0384e+00, -4.5539e-02,
        6.4847e-01,  5.2389e-01,  2.1804e-01,  6.2524e-02,  6.4810e-01,
        -3.9373e-02, -8.0147e-01,  7.3351e-01,  1.1177e+00,  2.1494e+00,
        -9.0878e-01, -6.7103e-01, -5.8041e-01, -8.5563e-02,  1.3945e+00,
        1.8056e-01,  1.3615e+00,  2.0372e+00,  6.4304e-01, -7.3257e-01,
        -4.8771e-01, -2.3396e-01,  7.0732e-01,  9.2781e-01,  4.8258e-01,
        -8.2979e-01,  1.2678e+00,  2.7356e-01, -6.1465e-01, -2.3494e-02,
        1.1717e+00,  1.5078e-01, -1.0411e+00, -7.2053e-01, -2.2148e+00,
        -6.8373e-01,  5.1636e-01,  5.5880e-01,  7.9176e-01,  4.2280e-01,
        -1.8687e+00, -1.1057e+00,  1.4373e-01,  5.8360e-01,  1.3482e+00,
        -8.1373e-01,  8.1999e-01, -1.3533e+00, -2.0710e-01, -2.4876e-01,
        -1.2320e+00,  6.2567e-01, -1.2231e+00, -6.2322e-01, -2.1625e-01,
        -7.8038e-01, -8.7388e-01, -7.3280e-01,  5.1430e-01,  1.0046e+00,
        -4.3352e-01,  1.1685e+00,  7.7037e-01,  3.9068e-01,  2.8959e-01,
        -2.7575e+00, -8.3236e-01,  1.4660e+00, -1.2191e+00, -1.1311e+00,
        -9.3218e-04, -1.6269e-01, -2.4772e-01,  2.4197e+00,  1.6456e+00}
        };
        conv.kernals[0] = new Matrix(1, 2, 2);
        conv.kernals[0].matrix = new Double[][]{{-0.0022,  0.1549, -0.2376, -0.2124}};
        conv.kernals[1] = new Matrix(1, 2, 2);
        conv.kernals[1].matrix = new Double[][]{{-0.1112,  0.0774, -0.0057,  0.2289}};
        conv.kernals[2] = new Matrix(1, 2, 2);
        conv.kernals[2].matrix = new Double[][]{{-0.0256,  0.0764,-0.0872, -0.0567}};
        conv.kernals[3] = new Matrix(1, 2, 2);
        conv.kernals[3].matrix = new Double[][]{{-0.2758, -0.1912,-0.1190,  0.0107}};
        conv.kernals[4] = new Matrix(1, 2, 2);
        conv.kernals[4].matrix = new Double[][]{{ 0.1141,  0.1732,-0.1957, -0.1257}};
        conv.kernals[5] = new Matrix(1, 2, 2);
        conv.kernals[5].matrix = new Double[][]{{0.1049,  0.2397,-0.0594,  0.2160}};
        conv.biases = new Matrix(1, 2, 1);
        conv.biases.matrix = new Double[][]{{-0.0465,  0.0305}};
        Matrix out = conv.forward(input, 0);
        Flatten flat = new Flatten();
        Matrix actual = flat.forward(out, 0);
        Matrix expected = new Matrix(1, 9 * 9 * 2, 1);
        expected.matrix = new Double[][]{{-0.2200,  0.4209,  0.3891,  0.5247,  0.1562,  0.3495, -0.2014,0.1352,  0.3869,-0.3524,  0.4762, -0.3580,  0.0781,  0.4595,  0.3204, -0.3139,0.4369, -0.0668,0.1266, -0.0281,  0.0219, -0.3503, -0.8508, -0.3736, -0.0594,0.4342, -0.0397,-0.0036,  0.2592,  0.1561,  0.6666, -0.2666, -0.2953,  0.0030,-0.3614, -1.1817,0.0595, -0.1816,  0.3106, -0.0189,  0.0076, -0.5597,  0.1968,0.4679,  0.3871,-0.2154, -0.4466, -0.6319, -0.9016, -0.4225, -0.8336, -0.6215,-0.1708,  0.0895,-0.3704,  0.5322, -0.1454, -0.0124, -0.1547,  0.6363,  0.0062,-0.1669, -0.3771,-0.8525,  0.7721,  0.5218,  0.2267,  0.0602,  0.4005,  0.6504,0.5556, -0.2787,0.3362,  0.5274, -0.0952,  0.3594, -0.5949, -0.1167,  0.4206,-0.1545,  0.1088,-0.1752, -0.0347,  0.4933,  0.2374, -0.6243, -0.2605, -0.7097,-1.3053, -0.6729,0.4036,  0.1611,  0.4553, -0.2491, -0.3209,  0.4474,  0.5935,-0.3583,  0.0476,-0.0668,  0.7819,  1.1413,  0.7632,  0.2232,  0.2744, -0.0411,0.6053,  0.7352,-0.2617,  0.4620,  0.6738,  0.3826, -1.0443, -0.8067,  0.8293,0.8309,  0.4130,0.9796,  1.3051,  1.1141,  0.2967,  0.1148, -0.7397,  0.4408,0.3586, -0.9063,0.4992,  0.2094,  0.4256, -0.2214, -0.1760,  0.6599, -0.1225,-0.6409, -0.6755,0.0073, -0.0027, -0.6996, -0.8825, -1.1005, -0.5838, -0.8581,0.2281,  1.0349,-0.0043,  0.2559, -0.4881, -0.1149, -0.2618,  0.5771, -0.4618,-0.5093, -0.2962,-1.3430, -0.2727,  0.1503,  0.9286,  1.1572,  1.4308,  1.2364,1.6413,  0.1275}};
        assert(actual.size == expected.size);
        for(int i = 0; i < actual.size; i++){
            assert(Math.abs(actual.matrix[0][i] - expected.matrix[0][i]) < 0.001);
        }
    }
}
// Parameter containing:
// tensor([-0.0465,  0.0305], requires_grad=True)

/*output
tensor([[[-0.2200,  0.4209,  0.3891,  0.5247,  0.1562,  0.3495, -0.2014,
0.1352,  0.3869],
[-0.3524,  0.4762, -0.3580,  0.0781,  0.4595,  0.3204, -0.3139,
0.4369, -0.0668],
[ 0.1266, -0.0281,  0.0219, -0.3503, -0.8508, -0.3736, -0.0594,
0.4342, -0.0397],
[-0.0036,  0.2592,  0.1561,  0.6666, -0.2666, -0.2953,  0.0030,
-0.3614, -1.1817],
[ 0.0595, -0.1816,  0.3106, -0.0189,  0.0076, -0.5597,  0.1968,
0.4679,  0.3871],
[-0.2154, -0.4466, -0.6319, -0.9016, -0.4225, -0.8336, -0.6215,
-0.1708,  0.0895],
-0.1545,  0.1088]],

[[-0.1752, -0.0347,  0.4933,  0.2374, -0.6243, -0.2605, -0.7097,
-1.3053, -0.6729],
[ 0.4036,  0.1611,  0.4553, -0.2491, -0.3209,  0.4474,  0.5935,
-0.3583,  0.0476],
[-0.0668,  0.7819,  1.1413,  0.7632,  0.2232,  0.2744, -0.0411,
0.6053,  0.7352],
[-0.2617,  0.4620,  0.6738,  0.3826, -1.0443, -0.8067,  0.8293,
0.8309,  0.4130],
[ 0.9796,  1.3051,  1.1141,  0.2967,  0.1148, -0.7397,  0.4408,
0.3586, -0.9063],
[ 0.4992,  0.2094,  0.4256, -0.2214, -0.1760,  0.6599, -0.1225,
-0.6409, -0.6755],
[ 0.0073, -0.0027, -0.6996, -0.8825, -1.1005, -0.5838, -0.8581,
0.2281,  1.0349],
[-0.0043,  0.2559, -0.4881, -0.1149, -0.2618,  0.5771, -0.4618,
-0.5093, -0.2962],
[-1.3430, -0.2727,  0.1503,  0.9286,  1.1572,  1.4308,  1.2364,
1.6413,  0.1275]]], grad_fn=<SqueezeBackward1>)*/
