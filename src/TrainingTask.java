package src;

import java.util.concurrent.Callable;

public class TrainingTask implements Callable<Integer> {
    private final int threadIndex;
    private final Model model;
    private final Image[][] batches;
    private int batchIndexForThread;
    private AtomicDouble avgLoss;

    public TrainingTask(int threadIndex, Model model, Image[][] batches, int batchIndexForThread, AtomicDouble avgLoss) {
        this.threadIndex = threadIndex;
        this.model = model;
        this.batches = batches;
        this.batchIndexForThread = batchIndexForThread;
        this.avgLoss = avgLoss;
    }

    public double trainImages() throws Exception {
        double loss = 0;
        for(int i = threadIndex; i < batches[batchIndexForThread].length; i += Main.numThreads){
            if(batches[batchIndexForThread][i] == null){
                break;
            }
            loss += model.forward(batches[batchIndexForThread][i].imageData, batches[batchIndexForThread][i].label, threadIndex);
            model.backward(threadIndex);
        }
        return loss;
    }

    public void setBatchIndex(int n) {
        batchIndexForThread = n;
    }

    @Override
    public Integer call() throws Exception {
        avgLoss.add(trainImages());
        return 1;
    }
}
