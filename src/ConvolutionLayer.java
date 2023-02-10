package src;

public class ConvolutionLayer {
    Matrix input;
    int NUM_FEATURE_SETS;
    int STRIDE;
    int KERNAL_SIZE;
    public Matrix[] kernals;
    public ConvolutionLayer(int NUM_FEATURE_SETS, int STRIDE, int KERNAL_SIZE){
        this.NUM_FEATURE_SETS = NUM_FEATURE_SETS;
        this.kernals = new Matrix[NUM_FEATURE_SETS];
        for(int i = 0; i < kernals.length; i++){
            kernals[i] = new Matrix(KERNAL_SIZE, KERNAL_SIZE);
            kernals[i].seed();
        }
        this.STRIDE = STRIDE;
        this.KERNAL_SIZE = KERNAL_SIZE;
    }
    
    public Matrix[] forward(Matrix input){
        Matrix[] result = new Matrix[NUM_FEATURE_SETS];
        Matrix featureSet = new Matrix(input.rows - KERNAL_SIZE + 1, input.cols - KERNAL_SIZE + 1);
        for(int i = 0 ; i < NUM_FEATURE_SETS; i++){
            featureSet = input.convolution(kernals[i]);
            result[i] = featureSet;        }
        return result;
    }
}
