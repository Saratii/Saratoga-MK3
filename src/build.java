package src;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class build {
    public static Model buildModel() throws FileNotFoundException, IOException{
        Model model = new Model();
        Pattern pattern = Pattern.compile("\\d+"); //lemme find where it used
        Matcher matcher;
        try (BufferedReader buildFile = new BufferedReader(new FileReader("logs/log-architecture"))) {
            String line;
            while((line = buildFile.readLine()) != null){
                String layerName = line.substring(line.lastIndexOf('.') + 1, line.indexOf('@'));
                if(layerName.equals("ConvolutionLayer")){
                    int numKernals = 0;
                    int stride = 0;
                    int kernalSize = 0;
                    Matrix[] kernals = new Matrix[0];
                    int kernalIndex = 0;
                    try (BufferedReader configFile = new BufferedReader(new FileReader("logs/log-" + line))) {
                        while((line = configFile.readLine()) != null){
                            if(line.contains("Number of Kernals")){
                                matcher = pattern.matcher(line);
                                if (matcher.find()){
                                    numKernals = Integer.parseInt(matcher.group());
                                    kernals = new Matrix[numKernals];
                                }
                            } else if(line.contains("Stride")){
                                matcher = pattern.matcher(line);
                                if (matcher.find()){
                                    stride = Integer.parseInt(matcher.group());
                                }
                            } else if(line.contains("Size of Kernals")){
                                matcher = pattern.matcher(line);
                                if (matcher.find()){
                                    kernalSize = Integer.parseInt(matcher.group());
                                }
                            } else if(line.contains("[")){
                                String s = line.replace("[", "").replace("]", "").replace(" ", "");
                                String[] strings = s.split(",");
                                Double[] doubleValues = Arrays.stream(strings).map(Double::valueOf).toArray(Double[]::new);
                                Matrix kernal = new Matrix(1, (int)Math.sqrt(doubleValues.length), (int)Math.sqrt(doubleValues.length));
                                kernal.matrix = new Double[][]{doubleValues};
                                kernals[kernalIndex] = kernal;
                                kernalIndex++;
                            }
                        }
                        ConvolutionLayer conv = new ConvolutionLayer(numKernals, stride, kernalSize);
                        conv.initialized = true;
                        conv.kernals = kernals;
                        model.layers.add(conv);
                    }
                } else if(layerName.equals("DenseLayer")){
                    int numNodes = 0;
                    int numInputs;
                    boolean biasFlag = true;
                    Matrix weights = new Matrix(0, 0, 0);
                    Matrix biases = new Matrix(0, 0 , 0);
                    try (BufferedReader configFile = new BufferedReader(new FileReader("logs/log-" + line))) {
                        while((line = configFile.readLine()) != null){
                            if(line.contains("Number of Nodes")){
                                matcher = pattern.matcher(line);
                                if (matcher.find()){
                                    numNodes = Integer.parseInt(matcher.group());
                                    biases = new Matrix(1, numNodes, 1);
                                }
                            } else if(line.contains("Number of Inputs")){
                                matcher = pattern.matcher(line);
                                if(matcher.find()){
                                    numInputs = Integer.parseInt(matcher.group());
                                    weights = new Matrix(1, numInputs, numNodes);
                                }

                            } else if(line.contains("[") && biasFlag == true){
                                String s = line.replace("[", "").replace("]", "").replace(" ", "");
                                String[] strings = s.split(",");
                                biases.matrix[0] = Arrays.stream(strings).map(Double::valueOf).toArray(Double[]::new);
                                biasFlag = false;
                            } else if(line.contains("[") && biasFlag == false){
                                String s = line.replace("[", "").replace("]", "").replace(" ", "");
                                String[] strings = s.split(",");
                                weights.matrix[0] = Arrays.stream(strings).map(Double::valueOf).toArray(Double[]::new);
                            }
                        }
                    }
                    DenseLayer dense = new DenseLayer(numNodes);
                    dense.initialized = true;
                    dense.weights = weights;
                    dense.biases = biases;
                    dense.outputs = new Matrix(1, numNodes, 1);

                    model.layers.add(dense);
                } else if(layerName.equals("Flatten")){
                    model.layers.add(new Flatten());
                } else if(layerName.equals("Softmax")){
                    model.layers.add(new Softmax());
                } else if(layerName.equals("MaxPool")){
                    int kernalSize = 0;
                    try (BufferedReader configFile = new BufferedReader(new FileReader("logs/log-" + line))) {
                        while((line = configFile.readLine()) != null){
                            if(line.contains("Size of Kernals")){
                                matcher = pattern.matcher(line);
                                if (matcher.find()){
                                    kernalSize = Integer.parseInt(matcher.group());
                                }
                            }
                        }
                    }
                    model.layers.add(new MaxPool(kernalSize));
                } else if(layerName.equals("ReLU")){
                    model.layers.add(new ReLU());
                }
            }
        }
        return model;
    }
}
