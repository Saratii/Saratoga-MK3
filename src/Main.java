package src;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_PURPLE = "\033[0;35m";
    public static double ALPHA = 0.001;
    public static double batchSize = 30;
    public static Model model;
    static double avgLoss = Double.POSITIVE_INFINITY;
    static Image[][] batches;
    static int batchIndexForThread;
    static int imageIndexForThread;
    final static int numThreads = 1;
    static int threadCounter = 0;
    static int chunkSize = 0;
    static int remainingData = 0;
    public static void main(String[] args) throws IOException {
        List<List<Image>> data = setupData(20, 0.0);
        List<Image> trainingData = data.get(0);
        List<Image> testingData = data.get(1);
        Boolean train = true;
        if(train){
            train(trainingData, 10);
        } else {
            Model builtModel  = build.buildModel();
            int numRight = 0;
            for(Image image: testingData){
                Matrix result = classify(image, builtModel);
                System.out.println("Predicted: " + result);
                System.out.println("Expected: " + image.label);
                System.out.println("\n");
                if((result.matrix[0][0] < result.matrix[0][1]) == (image.label.matrix[0][0] < image.label.matrix[0][1])){
                    numRight++;
                }
            }
            System.out.println("Percentage Correct: " + ((float) numRight / testingData.size()));
            System.out.println("Total Correct: " + numRight + " out of: " + testingData.size());
        }
        // ima smack you with my pimp cane
        // goofy ahh
    }
    public static void train(List<Image> trainingData, int batchSize) throws IOException{
        File folder = new File("logs");
        if(folder.exists()) { 
            String[] entries = folder.list();
            for(String s: entries){
                File currentFile = new File(folder.getPath(), s);
                currentFile.delete();
            }
        } else {
            folder.mkdirs(); 
        }
        PrintWriter writer = new PrintWriter("logs/log-graph", "UTF-8");
        
        
        batches = new Image[trainingData.size() / batchSize][batchSize];
        for(int i = 0; i < batches.length; i++){
            for(int j = 0; j < batches[0].length; j++){
                batches[i][j] = trainingData.get(i * batches[0].length + j);
            }
        }
    
        model = new Model();
        model.layers.add(new ConvolutionLayer(1, 12, 1, 3));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(3));
        model.layers.add(new ConvolutionLayer(12, 12, 1, 3));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(3));
        model.layers.add(new Flatten());
        model.layers.add(new DenseLayer(128));
        model.layers.add(new ReLU()); 
        model.layers.add(new DenseLayer(128));
        model.layers.add(new ReLU());
        model.layers.add(new DenseLayer(2));
        model.layers.add(new Softmax());
        model.profiling = false;
        int epoch = 0;
        ALPHA /= batchSize;
        
        long startTime = System.currentTimeMillis();

        
        Thread[] threads = new Thread[numThreads];
         // num_threads
       

        while(avgLoss > 0.01 && epoch < 5){
            avgLoss = 0;
            for(int i = 0; i < batches.length; i++){
                chunkSize = Math.floorDiv(batches[i].length, numThreads); // floor division
                remainingData = batches[i].length % numThreads; // remainder
                batchIndexForThread = i;
                for(threadCounter = 0; threadCounter < numThreads; threadCounter++) {
                    threads[threadCounter] = new Thread(() -> trainImages());
                    threads[threadCounter].start();
                }
                for (int j = 0; j < numThreads; j++) {
                    try {
                        threads[j].join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                model.updateParams();
            }
            avgLoss /= (batches[0].length * batches.length);
            System.out.println(ANSI_GREEN + "Epoch: " + epoch + " Average Loss: " + avgLoss + ANSI_RESET);
            writer.println(epoch + ", " + avgLoss);
            epoch++;
        }
        System.out.println(ANSI_CYAN + "Completed in " + epoch + " epochs" + ANSI_RESET);
        System.out.println(ANSI_CYAN + "Average time per epoch: " + ((System.currentTimeMillis() - startTime) / epoch) + " ms" + ANSI_RESET);
        if(model.profiling){
            for(int i = 0; i < model.layers.size(); i++){
                System.out.println("Forward: " + model.layers.get(i).forwardTime);
                System.out.println("Backward: " + model.layers.get(i).backwardTime + "\n");
            }
        }
        writer.close();
        model.write();
    }
    public static void trainImages(){
        for(int i = threadCounter * chunkSize + Math.min(threadCounter, remainingData); i < (threadCounter + 1) * chunkSize + Math.min(i + 1, remainingData) - 1; i++){
            avgLoss += model.forward(batches[batchIndexForThread][i].imageData, batches[batchIndexForThread][i].label);
            model.backward();
        }
    }
    public static Matrix classify(Image im, Model model) throws FileNotFoundException, IOException{
        model.forward(im.imageData, im.label);
        Softmax soft = (Softmax) model.layers.get(model.layers.size() - 1);
        return soft.result;
    }
    public static List<List<Image>> setupData(int imagesPerClass, double percentageTested) throws IOException{
        List<Path> dogDirectory = Files.list(Path.of("Animals/dog")).limit(imagesPerClass).toList();
        List<Path> elefanteDirectory = Files.list(Path.of("Animals/elefante")).limit(imagesPerClass).toList();
        List<Image> images = new ArrayList<>();
        List<Image> testImages = new ArrayList<>();
        for(int i = 0; i < dogDirectory.size(); i++){
            if(i < (1 - percentageTested) * dogDirectory.size()){
                images.add(new Image(dogDirectory.get(i), "dog"));
            } else {
                testImages.add(new Image(dogDirectory.get(i), "dog"));
            }
        }

        for(int i = 0; i < elefanteDirectory.size(); i++){
            if(i < (1 - percentageTested) * elefanteDirectory.size()){
                images.add(new Image(elefanteDirectory.get(i), "elefante"));
            } else {
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
}

