/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author gifarikautsar
 */
public class ANN {
    public enum WeightOptions {
        RANDOM,
        GIVEN
    }
    public enum TopologyOptions {
        MULTI_LAYER_PERCEPTRON,
        PERCEPTRON_TRAINING_RULE,
        DELTA_RULE_BATCH,
        DELTA_RULE_INCREMENTAL
    }
    public enum ActivationFunctionOptions {
        STEP,
        SIGN,
        SIGMOID
    }
   
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        Scanner sc = new Scanner(System.in);
        ANNOptions annOptions = new ANNOptions();
        String datasetURL;
        
//        System.out.println("Initial Weight");
//        System.out.println("1. Random");
//        System.out.println("2. Given");
//        annOptions.weightOpt = sc.nextInt();
//        
//        System.out.println("Topology");
//        System.out.println("1. Perceptron Training Rule");
//        System.out.println("2. Delta Rule - Batch");
//        System.out.println("3. Delta Rule - Incremental");
//        System.out.println("4. Multi Layer Perceptron");
//        annOptions.topologyOpt = sc.nextInt();
//        
//        if(annOptions.topologyOpt == 4){ // Multi Layer Perceptron
//            System.out.print("Hidden Layer: ");
//            annOptions.hiddenLayer = sc.nextInt();
//            for (int i = 0 ; i < annOptions.hiddenLayer ; i++) {
//                System.out.print("Neuron in Layer " + i+1 + ": ");
//                int nNeuron = sc.nextInt();
//                annOptions.layerNeuron.add(nNeuron);
//            }
//            System.out.print("Momentum: ");
//            annOptions.momentum = sc.nextDouble();
//        }
//        
//        System.out.println("Activation Function");
//        System.out.println("1. Step");
//        System.out.println("2. Sign");
//        System.out.println("3. Sigmoid");
//        
//        annOptions.activationFunctionOpt = sc.nextInt();
//        
//        System.out.print("Learning Rate: ");
//        annOptions.learningRate = sc.nextDouble();
//        
//        System.out.print("Threshold: ");
//        annOptions.threshold = sc.nextDouble();
//        
//        System.out.print("MaxIteration: ");
//        annOptions.maxIteration = sc.nextInt();
//        
//        System.out.println("Dataset URL: ");
//        datasetURL = sc.next();
        datasetURL = "dataset/[1].arff";
        System.out.println(datasetURL);
        Instances data = loadDataset(datasetURL);
        data.setClassIndex(data.numAttributes()-1);        
                
        if(annOptions.topologyOpt < 4){ // Perceptron Training Rule
            try {
                SingleLayerPerceptron perceptronTrainingRule  = new SingleLayerPerceptron(annOptions);
                perceptronTrainingRule.buildClassifier(data);
            } catch (Exception ex) {
                Logger.getLogger(ANN.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        else{ // Multi Layer Perceptron
            
        }
    }
    
    public static Instances loadDataset(String url){
        Instances dataset = null;
        try {
            System.out.println(url);
            dataset = ConverterUtils.DataSource.read(url);
        } catch (Exception ex) {
            System.out.println("File tidak berhasil di-load");
        }
        return dataset;
    }
}
