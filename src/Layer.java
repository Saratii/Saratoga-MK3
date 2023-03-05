package src;

public abstract class Layer {
    boolean firstBatch = true;
    public Matrix forward(Matrix inputs){
        return inputs;
    }
    public Matrix backward(Matrix inputs){
        return inputs;
    }
    public abstract void updateParams();
}
