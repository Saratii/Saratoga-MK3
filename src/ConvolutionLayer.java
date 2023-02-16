package src;

public class ConvolutionLayer {
    Matrix input;
    int NUM_FEATURE_SETS;
    int STRIDE;
    int KERNAL_SIZE;
    public Matrix[] kernals;
    public ConvolutionLayer(int NUM_FEATURE_SETS, int STRIDE, int KERNAL_SIZE){
        this.NUM_FEATURE_SETS = NUM_FEATURE_SETS;
        this.STRIDE = STRIDE;
        this.KERNAL_SIZE = KERNAL_SIZE;
    }
    
    public Matrix forward(Matrix input){
        this.kernals = new Matrix[NUM_FEATURE_SETS * input.z];
        for(int i = 0; i < kernals.length; i++){
            kernals[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE);
            kernals[i].seed();
        }
        Matrix result = new Matrix(NUM_FEATURE_SETS * input.z, input.rows - KERNAL_SIZE + 1, input.cols - KERNAL_SIZE + 1);
        for(int i = 0 ; i < result.matrix.length; i++){
            result.matrix[i] = input.convolution(kernals[i]).matrix[0];
        }
        return result;
    }
}
