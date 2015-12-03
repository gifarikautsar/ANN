package ann;


import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import weka.core.Instances;
import weka.core.SerializationHelper;
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
public class ANNOptions implements Serializable{
    public int weightOpt, topologyOpt, activationFunctionOpt, maxIteration, hiddenLayer;
    public double learningRate, threshold, momentum;
    public List<Integer> layerNeuron;
    private static String savedConfiguration = "ANNOptions.config";
    public List<Neuron> output;
    public List<List<Neuron>> layer;
    private Normalize normalize;
    private NominalToBinary ntb;
    
    public ANNOptions() {
        normalize = new Normalize();
        ntb = new NominalToBinary();
        output = new ArrayList<Neuron>();
        layer = new ArrayList<List<Neuron>>();
        layerNeuron = new ArrayList<Integer>();
        
        weightOpt = 2;
        topologyOpt = 4;
        activationFunctionOpt = 1;
        hiddenLayer = 1;
        layerNeuron.add(3);
        maxIteration = 1000;
        momentum = 0.2;
        learningRate = 0.3;
        threshold = 0.01;
    }
    public ANNOptions loadConfiguration(){
        ANNOptions annOptions = null;
        try {
            FileInputStream fileIn = new FileInputStream(savedConfiguration);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            annOptions = (ANNOptions) in.readObject();
            in.close();
            fileIn.close();

        } catch (Exception ex) {
            System.out.println(ex.toString());
        }
        return annOptions;
    }
    
    public void saveConfiguration(ANNOptions annOptions){
        try {
            FileOutputStream fileOut = new FileOutputStream(savedConfiguration);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(annOptions);
            out.close();
            fileOut.close();
        } catch (Exception ex) {
            System.out.println(ex.toString());
        }
    }
    
    public void initWeightsSLP(Instances data) throws Exception{
        ntb.setInputFormat(data);
        data = new Instances(Filter.useFilter(data, ntb));
        
        //normalize filter
        normalize.setInputFormat(data);
        data = new Instances(Filter.useFilter(data, normalize));
        
        int nAttr = data.numAttributes();
        Scanner sc = new Scanner(System.in);
        int nOutput;
        if(data.numClasses()<=2 && topologyOpt == 1){
            nOutput = 1;
        }
        else{
            nOutput = data.numClasses();
        }
        
        for(int j = 0; j<nOutput; j++){
            Neuron temp = new Neuron();
            if(weightOpt == 1){ // Random
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
    }
    
    public void initWeightsMLP(Instances data) throws Exception{
        ntb.setInputFormat(data);
        data = new Instances(Filter.useFilter(data, ntb));
        
        //normalize filter
        normalize.setInputFormat(data);
        data = new Instances(Filter.useFilter(data, normalize));
        
        int nAttr = data.numAttributes();
        Scanner sc = new Scanner(System.in);

        int nOutput = data.numClasses();
        
        for(int i = 0; i<hiddenLayer; i++){
            if(weightOpt == 2){
                System.out.println("Layer-" + (i+1));
            }
            List<Neuron> neuronLayer = new ArrayList<Neuron>();
            for(int j = 0; j<layerNeuron.get(i)+1; j++){
                if(weightOpt == 2)
                if(weightOpt == 2){
                    System.out.println("Neuron-" + (j+1));
                }
                Neuron neuron = new Neuron();
                if(i==0){ // weight from input layer
                    for(int k = 0; k < nAttr; k++){
                        if (weightOpt == 1) { // random 
                            Random random = new Random();
                            neuron.weights.add(random.nextDouble());
//                            neuron.weights.add(0.0);
                        } else { // given
                            if(k < nAttr-1){
                                if(weightOpt == 2){
                                    System.out.print("Weight input-" + (k+1) + ": ");
                                }
                            }
                            else{
                                if(weightOpt == 2){
                                    System.out.print("Weight bias: ");
                                }
                            }
                            neuron.weights.add(sc.nextDouble());
                        }
                    }
                    neuronLayer.add(neuron);
                }
                else if(j < layerNeuron.get(i)){ // weight from hidden layer
                    for(int k = 0; k < layerNeuron.get(i-1)+1; k++){ // layer neuron + 1, 1 for bias
                        if (weightOpt == 1) { // random 
                            Random random = new Random();
                            neuron.weights.add(random.nextDouble());
//                            neuron.weights.add(0.0);
                        } else { // given
                            if(k < layerNeuron.get(i-1)){
                                if(weightOpt == 2){
                                    System.out.print("Weight neuron-" + (k+1) + ": ");
                                }
                            }
                            else{
                                if(weightOpt == 2){
                                    System.out.print("Weight bias: ");
                                }
                            }
                            neuron.weights.add(sc.nextDouble());
                        }
                    }
                    neuronLayer.add(neuron);
                }
                
            }
            if(i!=0){
                Neuron bias = new Neuron();
                neuronLayer.add(bias);
            }
            layer.add(neuronLayer);
        }
        
        //last hidden layer to output
        List<Neuron> neuronLayer = new ArrayList<Neuron>();
        for(int i = 0; i < nOutput; i++){
            Neuron neuron = new Neuron();
            for(int j = 0; j < layerNeuron.get(layerNeuron.size()-1)+1; j++){
                if (weightOpt == 1) { // random 
                    Random random = new Random();
//                    neuron.weights.add(random.nextDouble());
                    neuron.weights.add(0.0);
                } else { // given
                    if(j < layerNeuron.get(layerNeuron.size()-1)){
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
    }
}
