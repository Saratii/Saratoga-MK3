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
    public static final double ALPHA = 0.0001;
    public static final double batchSize = 30;
    public static final int imagesUsedPerClass = 400;
    public static final double percentageTested = 0.2;
    public static final int numThreads = 1;
    public static final Boolean train = true;
    public static final Boolean forceTest = true;
    public static final int maxEpochs = 200;

    
    public static void main(String[] args) throws IOException {
        List<List<Image>> data = setupData(imagesUsedPerClass, percentageTested);
        List<Image> trainingData = data.get(0);
        List<Image> testingData = data.get(1);
        Model model;
        if(train){
            model = train(trainingData, 10);
        } else {
            model = build.buildModel();
        }
        if(forceTest){
            int numRight = 0;
            for(Image image: testingData){
                Matrix result = classify(image, model);
                System.out.println("Predicted: " + result);
                System.out.println("Expected: " + image.label + "\n");
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
    public static Model train(List<Image> trainingData, int batchSize) throws IOException{
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
    
        Image[][] batches = new Image[Math.ceilDiv(trainingData.size(),batchSize)][batchSize];
        for(int i = 0; i < batches.length; i++){
            for(int j = 0; j < batches[i].length; j++){
                if(trainingData.size() <= i * batches[0].length + j){
                    break;
                }
                batches[i][j] = trainingData.get(i * batches[0].length + j);
            }
        } 
    
        Model model = new Model();
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
        double avgLoss = Double.POSITIVE_INFINITY;
        long startTime = System.currentTimeMillis();
        while(avgLoss > 0.01 && epoch < maxEpochs){
            avgLoss = 0;
            for(int i = 0; i < batches.length; i++){
                for(int j = 0; j < batches[i].length; j++){
                    avgLoss += model.forward(batches[i][j].imageData, batches[i][j].label, 0);
                    model.backward(0);
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
        model.write(model);
        return model;
    }
    public static Matrix classify(Image im, Model model) throws FileNotFoundException, IOException{
        model.forward(im.imageData, im.label, 0);
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

