package ann;

import java.io.Serializable;
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
    implements TechnicalInformationHandler, Sourcable, Serializable {
    
    
    // Attributes
    private ANNOptions annOptions;
    private List<Neuron> output;
    private Normalize normalize;
    private NominalToBinary ntb;
    
    // constructor
    
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
        annOptions = new ANNOptions();
        annOptions = annOptions.loadConfiguration();
        output = new ArrayList<Neuron>();
        normalize = new Normalize();
        ntb = new NominalToBinary();
        output = annOptions.output;
        
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        //nominal to binary filter
        ntb.setInputFormat(data);
        data = new Instances(Filter.useFilter(data, ntb));
        
        //normalize filter
        normalize.setInputFormat(data);
        data = new Instances(Filter.useFilter(data, normalize));
        
        // do main function
        doPerceptron(data);        
    }

    public void printWeight(){
        for(int i = 0; i < output.size(); i++){
            System.out.println("Output-" + i);
            for(int j = 0; j < output.get(i).weights.size(); j++){
                System.out.println("Neuron-" + j + ": " + output.get(i).weights.get(j));
            }
        }
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
                            if (i == data.numInstances()-1) { // update weight
                                output.get(j).weights.set(k, annOptions.learningRate*(weight+deltaWeightUpdate[k]));
                            }
                        }
                        else {
                            deltaWeight = annOptions.learningRate * (target - newOutput) * input;
                            output.get(j).weights.set(k, weight+deltaWeight);
                        }
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
            // Convergent
            if(errorEpoch <= annOptions.threshold){
                break;
            }
        }
    }
    
    public int[] classifyInstances(Instances data) throws Exception{
        int[] classValue = new int[data.numInstances()];
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        //nominal to binary filter
        ntb.setInputFormat(data);
        data = new Instances(Filter.useFilter(data, ntb));
        int right = 0;
        
        
        for(int i = 0; i < data.numInstances(); i++){
            int outputSize =output.size();
            double[] result = new double[outputSize];
            for(int j = 0; j < outputSize; j++){
                result[j] = 0.0;
                for(int k = 0; k < data.numAttributes(); k++){
                    double input = 1;
                    if(k < data.numAttributes()-1){
                        input = data.instance(i).value(k);
                    }
                    result[j] += output.get(j).weights.get(k) * input;
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
    
    public double classifyInstance(Instance inst) throws Exception{
        double instanceClass = 0.0;
        
        ntb.input(inst);
        inst = ntb.output();
        normalize.input(inst);
        inst = normalize.output();
        System.out.println(inst.toString());
        int outputSize =output.size();
        double[] result = new double[outputSize];
        for(int j = 0; j < outputSize; j++){
            result[j] = 0.0;
            for(int k = 0; k < inst.numAttributes(); k++){
                double input = 1;
                if(k < inst.numAttributes()-1){
                    input = inst.value(k);
                }
                System.out.println(output.get(j).weights.get(k) + "*" + input);
                result[j] += output.get(j).weights.get(k) * input;
            }
            result[j] = Util.activationFunction(result[j], annOptions);
        }

        if(outputSize >= 2){
            for(int j = 0; j < outputSize; j++){
                System.out.print(result[j] + "-");
                if(result[j] > result[(int)instanceClass]){
                    
                    instanceClass  = j;
                    System.out.println(instanceClass);
                }
            }
        }
        else{
            instanceClass = (int)result[0];
        }
        
        return instanceClass;
    } 
}
