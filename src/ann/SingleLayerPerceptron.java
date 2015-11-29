package ann;

import static java.lang.Math.exp;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author gifarikautsar
 */
public class SingleLayerPerceptron 
    extends Classifier 
    implements TechnicalInformationHandler, Sourcable {
    
    
    // Attributes
    private ANNOptions annOptions;
    private List<Neuron> output;
    
    // constructor
    public SingleLayerPerceptron(ANNOptions annOptions_) throws Exception{
        annOptions = annOptions_;
        output = new ArrayList<Neuron>();
    }
    
    public void initWeights(Instances data){
        int nAttr = data.numAttributes();
        Scanner sc = new Scanner(System.in);
        int nOutput;
        if(data.numClasses()<=2 && annOptions.topologyOpt == 1){
            nOutput = 1;
        }
        else{
            nOutput = data.numClasses();
        }
        
        for(int j = 0; j<nOutput; j++){
            Neuron temp = new Neuron();
            if(annOptions.weightOpt == 1){ // Random
                for(int i = 0; i < nAttr; i++) {
                    Random random = new Random();
                    temp.weights.add(random.nextDouble());
//                    temp.weights.add(0.0);
                } 
            }
            else{ // Given
                System.out.println("Output-" + j);
                for(int i = 0; i < nAttr-1; i++) {
                    System.out.print("Weight-" + (i+1) + ": ");
                    temp.weights.add(sc.nextDouble());
                }
                System.out.print("Bias weight: ");
                temp.weights.add(sc.nextDouble());
            }
            
            output.add(temp);
        }
        
//        for(int j = 0; j < nOutput; j++){
//            for(int i = 0; i < nAttr; i++) {
//                System.out.println(output.get(j).weights.get(i));
//            } 
//        }
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        data = Util.setNominalToBinary(data);
        data = Util.setNormalize(data);
        initWeights(data);
        // do main function
        doPerceptron(data);
        
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String toSource(String string) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    // Main Function
    public void doPerceptron(Instances data) { 
        for(int epoch = 0; epoch<annOptions.maxIteration; epoch++){
            double deltaWeight = 0.0;
            double[] deltaWeightUpdate = new double[data.numAttributes()];
            for(int i = 0; i < data.numAttributes(); i++){
                deltaWeightUpdate[i] = 0;
            }
            for(int i = 0; i < data.numInstances(); i++){
                // do sum xi.wi (nilai data * bobot)
                for(int j = 0; j < output.size(); j++){
                    double sum = 0;
                    double weight, input;
                    for(int k = 0; k < data.numAttributes(); k++){
                        if(k == data.numAttributes()-1){ // bias
                            input = 1;
                        }
                        else{
                            input = data.instance(i).value(k);                    
                        }
                        weight = output.get(j).weights.get(k);
    //                    System.out.println("Weight: " + weight);
    //                    System.out.println("Input: " + input);
                        sum += weight * input;
                    }

                    // Update input weight
                    for(int k = 0; k < data.numAttributes(); k++){
                        if(k == data.numAttributes()-1){ // bias
                            input = 1;
                        }
                        else{
                            input = data.instance(i).value(k);                    
                        }

                        // lewati fungsi aktivasi
                        double newOutput = Util.activationFunction(sum,annOptions);
                        double target;
                        if(output.size() > 1){
                            if(data.instance(i).classValue() == j){
                                target = 1;
                            }
                            else{
                                target = 0;
                            }
                        }
                        else{
                            target = data.instance(i).classValue();
                        }
                        weight = output.get(j).weights.get(k);

                        // hitung delta weight -> learning rate * (T-O) * xi
                        if (annOptions.topologyOpt == 2) // batch
                        {
                            deltaWeightUpdate[k] += (target - newOutput) * input;
//                            System.out.println("deltaWeight: " + ((target - newOutput) * input));
                            if (i == data.numInstances()-1) { // update weight
                                output.get(j).weights.set(k, annOptions.learningRate*(weight+deltaWeightUpdate[k]));
                                System.out.println("weight: " + (annOptions.learningRate*(weight+deltaWeightUpdate[k])));
                            }
                        }
                        else {
                            deltaWeight = annOptions.learningRate * (target - newOutput) * input;
                            output.get(j).weights.set(k, weight+deltaWeight);
                        }
                            
//                        System.out.println("---");
//                        System.out.println("target: " + target);
//                        System.out.println("newOutput: " + newOutput);

                        // hitung bobot baru (bobot awal + delta bobot)
//                        System.out.println("Instance-" + (i+1) +" : " + output.get(j).weights.get(k));
    //                    System.out.println("Delta: " + deltaWeight);
    //                    System.out.println("-----");
                    }
                }   
            }

            // hitung error
            double errorEpoch = 0;
            for(int i = 0; i < data.numInstances(); i++){
                double sum = 0;
                for(int j = 0; j < output.size(); j++){
                    for(int k = 0; k < data.numAttributes(); k++){
                        double input;
                        if(k == data.numAttributes()-1){ // bias
                            input = 1;
                        }
                        else{
                            input = data.instance(i).value(k);                    
                        }
                        double weight = output.get(j).weights.get(k);
    //                    System.out.println("Weight: " + weight);
    //                    System.out.println("Input: " + input);
                        sum += weight * input;
                    }
                    // lewati fungsi aktivasi
                    sum = Util.activationFunction(sum,annOptions);
                    double target;
                    if(output.size() > 1){
                        if(data.instance(i).classValue() == j){
                            target = 1;
                        }
                        else{
                            target = 0;
                        }
                    }
                    else{
                        target = data.instance(i).classValue();
                    }
                    double error = target - sum;
                    errorEpoch += error * error;
                }
            }
            errorEpoch *= 0.5; 
//            System.out.println((epoch+1) + " : " + errorEpoch);
            // Convergent
            if(errorEpoch <= annOptions.threshold){
                break;
            }
        }
        System.out.println("DONE :)");
//        for(int i = 0; i < output.size(); i++){
//            System.out.println("---");
//            for(int j = 0; j < output.get(i).weights.size(); j++){
//                System.out.println(output.get(i).weights.get(j));
//            }
//        }
    }
    
    public int[] classifyInstances(Instances data){
        int[] classValue = new int[data.numInstances()];
        data = Util.setNominalToBinary(data);
        data = Util.setNormalize(data);
        int right = 0;
        
        
        for(int i = 0; i < data.numInstances(); i++){
            int outputSize =output.size();
            double[] result = new double[outputSize];
            for(int j = 0; j < outputSize; j++){
                result[j] = 0.0;
                for(int k = 0; k < data.numAttributes(); k++){
                    double input = 1;
                    if(k < data.numAttributes()-1){
                        input = output.get(j).weights.get(k);
                    }
                    result[j] += output.get(j).weights.get(k) * data.instance(i).value(k);
                }
                result[j] = Util.activationFunction(result[j], annOptions);
            }

            if(outputSize >= 2){
                for(int j = 0; j < outputSize; j++){
                    if(result[j] > result[classValue[i]]){
                        classValue[i]  = j;
                    }
                }
            }
            else{
                classValue[i] = (int)result[0];
            }
            double target = data.instance(i).classValue();
            double output = classValue[i];
            System.out.println("Intance-" + i + " target: " + target + " output: " + output);
            if(target == output){
                right = right + 1;
            }
        }
        
        System.out.println("Percentage: " + ((double)right/(double)data.numInstances()));
        
        return classValue;
    }
}
