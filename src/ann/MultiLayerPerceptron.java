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
    
    public MultiLayerPerceptron(ANNOptions annOptions_) {
        annOptions = annOptions_;
        layer = new ArrayList<List<Neuron>>();
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
        
        for(int i = 0; i<nOutput; i++){
            Neuron neuron = new Neuron();
            // input weight
            for(int j = 0; j < nAttr-1; j++) {
                if (annOptions.weightOpt == 1) { // random 
                    Random random = new Random();
                    neuron.weights.add(random.nextDouble());
                } else { // given
                    System.out.print("Weight input-" + (j+1) + ": ");
                    neuron.weights.add(sc.nextDouble());
                }
            } 
            System.out.println("Bias: ");
            neuron.weights.add(sc.nextDouble());
            layer.add(neuron);

            // weight hidden layer
            for(int j = 0 ;j<annOptions.hiddenLayer;j++) {
                neuron = new Neuron();
                System.out.println("Hidden Layer-" + (j+1));
                for (int k = 0 ; k < annOptions.layerNeuron.get(j);k++) {
                    if (annOptions.weightOpt == 1) { // random 
                        Random random = new Random();
                        neuron.weights.add(random.nextDouble());
                    } else { // given
                        System.out.print("Weight neuron-" + (k+1) + ": ");
                        neuron.weights.add(sc.nextDouble());
                    }
                }
                System.out.println("Bias: ");
                neuron.weights.add(sc.nextDouble());
                layer.add(neuron);
                System.out.println("-----");
            }
        }
        
        for(int i = 0; i < layer.size(); i++){
            System.out.println("Layer-" + (i));
            for (int j=0;j<layer.get(i).weights.size();j++) {
                System.out.println("Neuron-" + (j+1)+": "+layer.get(i).weights.get(j));
            }
        }
    }
    
    public void doPerceptron(Instances data) {
        int numAttr = data.numAttributes();
        int numInstances = data.numInstances();
        for (int epoch = 0 ; epoch <annOptions.maxIteration;epoch++) {
            for (int i = 0  ; i < numInstances;i++) {
                // input to layer 1
                for (int j = 0 ; j < annOptions.layerNeuron.get(0);j++) { // nNeuron in layer 1
                    double sum = 0;
                    for (int k = 0 ; k < numAttr;k++) {
                        double input;
                        if (k == numAttr-1) {
                            input = 1;
                        }
                        else {
                            input = data.instance(i).value(k);
                        }
                        double weight = layer.get(1).weights.get(k);
                        sum+= input * weight;
                    }
                    layer.get(1).value = sum;
                }
                
                
                for (int j = 1 ; j < layer.size();j++) {
                    
                }
            }
        }
    }
}
