/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.io.Serializable;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author gifarikautsar
 */
public class ANN implements Serializable{
   
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        Scanner sc = new Scanner(System.in);
        ANNOptions annOptions = new ANNOptions();
        String datasetURL;
        
//        System.out.println("Topology");
//        System.out.println("1. Perceptron Training Rule");
//        System.out.println("2. Delta Rule - Batch");
//        System.out.println("3. Delta Rule - Incremental");
//        System.out.println("4. Multi Layer Perceptron");
//        annOptions.topologyOpt = sc.nextInt();
        
//        System.out.println("Initial Weight");
//        System.out.println("1. Random");
//        System.out.println("2. Given");
//        annOptions.weightOpt = sc.nextInt();
        
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
        
        datasetURL = "dataset/weather.nominal.arff";
//        datasetURL = "dataset/weather.numeric.arff";
//        datasetURL = "dataset/iris.arff";
        
        Instances data = loadDataset(datasetURL);
        
        Classifier model = null;
        data.setClassIndex(data.numAttributes()-1);        
        if(annOptions.topologyOpt < 4){ // Perceptron Training Rule
            annOptions.initWeightsSLP(data);
            annOptions.saveConfiguration(annOptions);
            try {
                SingleLayerPerceptron singleLayerPerceptron  = new SingleLayerPerceptron();
                singleLayerPerceptron.buildClassifier(data);
                model = singleLayerPerceptron;
                crossValidation(model, data);
            } catch (Exception ex) {
                Logger.getLogger(ANN.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        else if (annOptions.topologyOpt == 4){ // Multi Layer Perceptron
            annOptions.initWeightsMLP(data);
            annOptions.saveConfiguration(annOptions);
            try {
                MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron();
                multiLayerPerceptron.buildClassifier(data);
                model = multiLayerPerceptron;
                crossValidation(model, data);
            } catch (Exception ex) {
                Logger.getLogger(ANN.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    public static Instances loadDataset(String url){
        Instances dataset = null;
        try {
            dataset = ConverterUtils.DataSource.read(url);
        } catch (Exception ex) {
            System.out.println("File tidak berhasil di-load");
        }
        return dataset;
    }
    
    public static void crossValidation(Classifier model, Instances data){
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, 10, new Random(1));
            System.out.println("================================");
            System.out.println("========Cross Validation========");
            System.out.println("================================");
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        } catch (Exception ex) {
            System.out.println(ex.toString());
        }
    }
    
    public void percentageSplit(Classifier model, double percent, Instances data){
        try {
            int trainSize = (int) Math.round(data.numInstances() * percent/100);
            int testSize = data.numInstances() - trainSize;
            Instances train = new Instances(data, trainSize);
            Instances test = new Instances(data, testSize);;
            
            for(int i=0; i<trainSize; i++){
                train.add(data.instance(i));
            }
            for(int i=trainSize; i<data.numInstances(); i++){
                test.add(data.instance(i));
            }
            
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(model, test);
            System.out.println("================================");
            System.out.println("========Percentage  Split=======");
            System.out.println("================================");
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        } catch (Exception ex) {
            System.out.println("File tidak berhasil di-load");
        }
    }
    
    public void saveModel(String modelname, Classifier model){
        try {
            SerializationHelper.write(modelname, model);
            System.out.println(modelname + " berhasil dibuat\n");
        } catch (Exception ex) {
            System.out.println(modelname + " tidak bisa dibuat\n");
        }
    }
    
    public Classifier loadModel(String modeladdress){
        Classifier model = null;
        try {
            model  = (Classifier) SerializationHelper.read(modeladdress);
            System.out.println(model.toString());
            System.out.println(modeladdress + " berhasil diload\n");
        } catch (Exception ex) {
            System.out.println(modeladdress + " tidak bisa diload\n");
        }
        return model;
    }
    
    public void classify(String data_address, Classifier model){
        try {
            Instances test = ConverterUtils.DataSource.read(data_address);
            test.setClassIndex(test.numAttributes()-1);
            System.out.println("====================================");
            System.out.println("=== Predictions on user test set ===");
            System.out.println("====================================");
            System.out.println("# - actual - predicted - distribution");
            for (int i = 0; i < test.numInstances(); i++) {
                double pred = model.classifyInstance(test.instance(i));
                double[] dist = model.distributionForInstance(test.instance(i));
                System.out.print((i+1) + " - ");
                System.out.print(test.instance(i).toString(test.classIndex()) + " - ");
                System.out.print(test.classAttribute().value((int) pred) + " - ");
                System.out.println(Utils.arrayToString(dist));
            }
            System.out.println("\n");
        } catch (Exception ex) {
            System.out.println("Tidak berhasil memprediksi hasil\n");
        }
    }
}
