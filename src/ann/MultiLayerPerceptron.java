/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

/**
 *
 * @author Tony
 */
public class MultiLayerPerceptron 
    extends Classifier 
    implements TechnicalInformationHandler, Sourcable{
    
    // attributes
    private ANNOptions annOptions;
    private List<List<Neuron>> layer;
    private List<Double> bias;
    public MultiLayerPerceptron(ANNOptions annOptions_) {
        annOptions = annOptions_;
        layer = new ArrayList<List<Neuron>>();
        bias = new ArrayList<Double>();
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        data = Util.setNominalToBinary(data);
        
        initWeights(data);
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
    
    public void initWeights(Instances data){
        int nAttr = data.numAttributes();
        Scanner sc = new Scanner(System.in);
        System.out.println(data.numAttributes());
        int nOutput;
        
        if(data.numClasses()<=2){ // binary class
            nOutput = 1;
        }
        else{ // multi class
            nOutput = data.numClasses();
        }
        
        for(int i = 0; i<annOptions.hiddenLayer; i++){
            System.out.println("Layer-" + (i+1));
            List<Neuron> neuronLayer = new ArrayList<Neuron>();
            for(int j = 0; j<annOptions.layerNeuron.get(i); j++){
                System.out.println("Neuron-" + (j+1));
                Neuron neuron = new Neuron();
                if(i==0){ // weight from input layer
                    for(int k = 0; k < nAttr; k++){
                        if (annOptions.weightOpt == 1) { // random 
                            Random random = new Random();
//                            neuron.weights.add(random.nextDouble());
                            neuron.weights.add(0.0);
                        } else { // given
                            if(k < nAttr-1){
                                System.out.print("Weight input-" + (k+1) + ": ");
                            }
                            else{
                                System.out.print("Weight bias: ");
                            }
                            neuron.weights.add(sc.nextDouble());
                        }
                    }
                }
                else{ // weight from hidden layer
                    for(int k = 0; k < annOptions.layerNeuron.get(i-1)+1; k++){
                        if (annOptions.weightOpt == 1) { // random 
                            Random random = new Random();
//                            neuron.weights.add(random.nextDouble());
                            neuron.weights.add(0.0);
                        } else { // given
                            if(k < annOptions.layerNeuron.get(i-1)){
                                System.out.print("Weight neuron-" + (k+1) + ": ");
                            }
                            else{
                                System.out.print("Weight bias: ");
                            }
                            neuron.weights.add(sc.nextDouble());
                        }
                    }
                }
                neuronLayer.add(neuron);
            }
            layer.add(neuronLayer);
            System.out.println("-------");
        }
        
        //last hidden layer to output
        List<Neuron> neuronLayer = new ArrayList<Neuron>();
        for(int i = 0; i < nOutput; i++){
            System.out.println("Output-" + i);
            Neuron neuron = new Neuron();
            for(int j = 0; j < annOptions.layerNeuron.get(annOptions.layerNeuron.size()-1)+1; j++){
                if (annOptions.weightOpt == 1) { // random 
                    Random random = new Random();
//                    neuron.weights.add(random.nextDouble());
                    neuron.weights.add(0.0);
                } else { // given
                    if(j < annOptions.layerNeuron.get(annOptions.layerNeuron.size()-1)){
                        System.out.print("Weight neuron-" + (j+1) + ": ");
                    }
                    else{
                        System.out.print("Bias: ");
                    }
                    neuron.weights.add(sc.nextDouble());
                }
            }
            neuronLayer.add(neuron);
        }
        layer.add(neuronLayer);
        System.out.println("-------");
        
        //print weight
//        for(int i = 0; i < layer.size(); i++){
//            System.out.println("Layer-" + (i+1));
//            for (int j=0;j<layer.get(i).size();j++) {
//                System.out.println("Neuron-" + (j+1));
//                for(int k = 0; k < layer.get(i).get(j).weights.size(); k++){
//                    System.out.println("Weight-" + (k+1) + ": " + layer.get(i).get(j).weights.get(k));
//                }
//                System.out.println("-------");
//            }
//        }
    }
    
    public void doPerceptron(Instances data) {
        int numAttr = data.numAttributes();
        int numInstances = data.numInstances();
        for (int epoch = 0; epoch <annOptions.maxIteration;epoch++) {
            for (int i = 0; i < numInstances;i++) {
                for(int j = 0; j < layer.size(); j++){ // Iterasi sebanyak jumlah layer (hidden layer + output layer)
                    for (int k = 0; k < layer.get(j).size(); k++){ // Iterasi sebanyak jumlah neuron pada layer (udah termasuk bias, bias = neuron terakhir pd layer)
                        double sum = 0.0;
                        if(j == 0){ // Untuk hidden layer pertama, value diambil dari input
                            for(int l = 0; l < numAttr; l++){                                
                                double input;
                                if(k == data.numAttributes()-1){ // bias
                                    input = 1;
                                }
                                else{
                                    input = data.instance(i).value(l);                    
                                }
                                double weight = layer.get(j).get(k).weights.get(l);
                                sum += input * weight;
                            }
                        }
                        else{ // Untuk hidden layer > 1, Value diambil dari neuron value pada layer sebelumnya
                            int nPrevLayer = layer.get(j-1).size();
                            for(int l = 0; l < nPrevLayer; l++){
                                double input;
                                if(k == nPrevLayer-1){ // bias
                                    input = 1;
                                }
                                else{
                                    input = layer.get(j-1).get(l).value;
                                }
                                double weight = layer.get(j).get(k).weights.get(l);
                                sum += input * weight;
                            }
                        }
                        
                        if (k < numAttr-1) { // if not bias
                            System.out.println("Before Layer-" + (j+i) + " neuron-" + (k+1) + ": " + sum);
                            sum = Util.activationFunction(sum, annOptions);
                            //sigmoid
                            layer.get(j).get(k).value = sum;
                        } 
                        
                        System.out.println("After Layer-" + (j+i) + " neuron-" + (k+1) + ": " + layer.get(j).get(k).value);
                    }
                }
                
                // let's calculate error guys
                // for output, madam
                int nOutputNeuron = layer.get(layer.size()-1).size();
                for (int x = 0; x< nOutputNeuron;x++) {
                    double output = layer.get(layer.size()-1).get(x).value;
                    double target = data.instance(i).classValue();
                    if(data.numClasses() > 2){
                        if(target == x){
                            target = 1;
                        }
                        else{
                            target = 0;
                        }
                    }
                    layer.get(layer.size()-1).get(x).error = calculateError(output,target);
                }
                
                for (int x = annOptions.hiddenLayer-1 ; x >= 0;x--) { // for each hidden layer
                    for (int y = 0; y<annOptions.layerNeuron.get(x);y++) { // for each neuron in hidden layer
                        double error = 0;
                    
                        for (int z = 0 ; z < layer.get(x+1).size();z++) { // for each neuron in next layer
                            double nextError = layer.get(x+1).get(z).error;
                            double weight = layer.get(x+1).get(z).weights.get(y);
                            error *= weight * nextError;
                        }
                        
                        double output = layer.get(x).get(y).value;
                        error *=  output * (1-output);
                    }
                }
            }
        }
    }
    
    public double calculateError(double output, double target){
        return (output *(1-output) *(target-output));
    }
    
    public double calculateError(double output, double errorAfter, double weight) {
        return (output * (1 - output) * errorAfter * weight);
    }
}
