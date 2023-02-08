package src;

public class ConvolutionLayer {
    Matrix input;
    int NUM_FEATURE_SETS;
    int STRIDE;
    int KERNAL_SIZE;
    public ConvolutionLayer(int NUM_FEATURE_SETS, int STRIDE, int KERNAL_SIZE){
        this.NUM_FEATURE_SETS = NUM_FEATURE_SETS;
        this.STRIDE = STRIDE;
        this.KERNAL_SIZE = KERNAL_SIZE;
    }
    
    public Matrix[] forward(Matrix input){
        Matrix[] result = new Matrix[NUM_FEATURE_SETS];
        Matrix featureSet = new Matrix(input.rows - KERNAL_SIZE + 1, input.cols - KERNAL_SIZE + 1);
        for(int i = 0 ; i < NUM_FEATURE_SETS; i++){
            Matrix kernal = new Matrix(KERNAL_SIZE, KERNAL_SIZE);
            kernal.seed();
            featureSet = input.convolution(kernal);
            result[i] = featureSet;
        }
        return result;
    }
}
