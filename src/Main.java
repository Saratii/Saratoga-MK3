package src;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

public class Main {
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_PURPLE = "\033[0;35m";
    public static final double ALPHA = 0.0001;
    public static final int batchSize = 20;
    public static final int imagesUsedPerClass = 300;
    public static final double percentageTested = 0.2;
    public static final int numThreads = 10;
    public static final Boolean train = true;
    public static final Boolean forceTest = true;
    public static final int maxEpochs = 2500;
    public static final int imageSize = 90;
    public static final double minimumLoss = 0.01;
    public static final boolean testWhileTrain = true;

    public static void main(String[] args) throws Exception {
        List<List<Image>> data = setupData(imagesUsedPerClass, percentageTested);
        List<Image> trainingData = data.get(0);
        List<Image> testingData = data.get(1);
        Model model;
        if(train){
            model = train(trainingData, batchSize, testingData);
        } else{
            model = build.buildModel();
        }
        if(forceTest){
            classifyAll(testingData, model);
        }
    }

    public static Model train(List<Image> trainingData, int batchSize) throws Exception {
        return train(trainingData, batchSize, null);
    }

    public static Model train(List<Image> trainingData, int batchSize, List<Image> testingData) throws Exception {
        String forGraph = "";
        Image[][] batches = new Image[Math.ceilDiv(trainingData.size(), batchSize)][batchSize];
        for(int i = 0; i < batches.length; i++){
            for(int j = 0; j < batches[i].length; j++){
                if(trainingData.size() <= i * batches[0].length + j){
                    break;
                }
                batches[i][j] = trainingData.get(i * batches[0].length + j);
            }
        }
        Model model = new Model();
        model.layers.add(new ConvolutionLayer(1, 24, 1, 5));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(3));
        model.layers.add(new ConvolutionLayer(24, 14, 1, 3));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(3));
        model.layers.add(new Flatten());
        model.layers.add(new DenseLayer(1134, 32));
        model.layers.add(new ReLU());
        model.layers.add(new Dropout(0.4));
        model.layers.add(new DenseLayer(32, 32));
        model.layers.add(new ReLU());
        model.layers.add(new Dropout(0.4));
        model.layers.add(new DenseLayer(32, 2));
        model.layers.add(new Softmax());
        int epoch = 0;
        AtomicDouble avgLoss = new AtomicDouble(Double.POSITIVE_INFINITY);
        long startTime = System.currentTimeMillis();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        TrainingTask[] threads = new TrainingTask[numThreads];
        Future<?>[] threadResults = new Future[numThreads];
        IntStream.range(0, numThreads).forEach(i -> threads[i] = new TrainingTask(i, model, batches, 0, avgLoss));
        while(avgLoss.get() > minimumLoss && epoch < maxEpochs){
            avgLoss.set(0.0);
            for(int i = 0; i < batches.length; i++){
                for(int j = 0; j < numThreads; j++){
                    threads[j].setBatchIndex(i);
                    threadResults[j] = executor.submit(threads[j]);
                }
                for(int j = 0; j < numThreads; j++){
                    try{
                        threadResults[j].get();
                    } catch (Exception e){
                        e.printStackTrace();
                        System.exit(1);
                    }
                }
                model.updateParams();
            }
            if(testWhileTrain){
                classifyAll(testingData, model);
            }
            avgLoss.divide(trainingData.size());
            System.out.println(ANSI_GREEN + "Epoch: " + epoch + " Average Loss: " + avgLoss + ANSI_RESET);
            forGraph = forGraph + epoch + ", " + avgLoss + "\n";
            epoch++;
        }
        executor.shutdown();
        System.out.println(ANSI_CYAN + "Completed in " + epoch + " epochs" + ANSI_RESET);
        System.out.println(ANSI_CYAN + "Average time per epoch: " + ((System.currentTimeMillis() - startTime) / epoch) + " ms" + ANSI_RESET);
        File folder = new File("logs");
        PrintWriter writer = new PrintWriter("logs/log-graph", "UTF-8");
        if(folder.exists()){
            String[] entries = folder.list();
            for(String s : entries){
                File currentFile = new File(folder.getPath(), s);
                currentFile.delete();
            }
        } else{
            folder.mkdirs();
        }
        writer.println(forGraph);
        writer.close();
        model.write();
        return model;
    }

    public static Matrix classify(Image im, Model model) throws Exception {
        model.forward(im.imageData, im.label, 0);
        Softmax soft = (Softmax) model.layers.get(model.layers.size() - 1);
        return soft.layerOutput[0];
    }

    public static List<List<Image>> setupData(int imagesPerClass, double percentageTested) throws IOException {
        List<Path> dogDirectory = Files.list(Path.of("Animals/dog")).limit(imagesPerClass).toList();
        List<Path> elefanteDirectory = Files.list(Path.of("Animals/elefante")).limit(imagesPerClass).toList();
        List<Image> images = new ArrayList<>();
        List<Image> testImages = new ArrayList<>();
        for(int i = 0; i < dogDirectory.size(); i++){
            if(i < (1 - percentageTested) * dogDirectory.size()){
                images.add(new Image(dogDirectory.get(i), "dog"));
            } else{
                testImages.add(new Image(dogDirectory.get(i), "dog"));
            }
        }
        for(int i = 0; i < elefanteDirectory.size(); i++){
            if(i < (1 - percentageTested) * elefanteDirectory.size()){
                images.add(new Image(elefanteDirectory.get(i), "elefante"));
            } else{
                testImages.add(new Image(elefanteDirectory.get(i), "elefante"));
            }
        }
        Image.shuffle(images);
        Image.shuffle(testImages);
        List<List<Image>> data = new ArrayList<>();
        data.add(images);
        data.add(testImages);
        return data;
    }

    public static void classifyAll(List<Image> testingData, Model model) throws Exception {
        int numRight = 0;
        model.isClassifying = true;
        for(Image image : testingData){
            Matrix result = classify(image, model);
            if((result.matrix[0][0] < result.matrix[0][1]) == (image.label.matrix[0][0] < image.label.matrix[0][1])){
                numRight++;
            }
        }
        model.isClassifying = false;
        System.out.println("Percentage Correct: " + ((float) numRight / testingData.size()));
        System.out.println("Total Correct: " + numRight + " out of: " + testingData.size());
    }
}