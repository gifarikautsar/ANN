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
//        System.out.println(data.numAttributes());
        int nOutput = data.numClasses();
        
        for(int i = 0; i<annOptions.hiddenLayer; i++){
            System.out.println("Layer-" + (i+1));
            List<Neuron> neuronLayer = new ArrayList<Neuron>();
            for(int j = 0; j<annOptions.layerNeuron.get(i)+1; j++){
                System.out.println("Neuron-" + (j+1));
                Neuron neuron = new Neuron();
                if(i==0){ // weight from input layer
                    for(int k = 0; k < nAttr; k++){
                        if (annOptions.weightOpt == 1) { // random 
                            Random random = new Random();
                            neuron.weights.add(random.nextDouble());
//                            neuron.weights.add(0.0);
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
                    neuronLayer.add(neuron);
                }
                else if(j < annOptions.layerNeuron.get(i)){ // weight from hidden layer
                    for(int k = 0; k < annOptions.layerNeuron.get(i-1)+1; k++){ // layer neuron + 1, 1 for bias
                        if (annOptions.weightOpt == 1) { // random 
                            Random random = new Random();
                            neuron.weights.add(random.nextDouble());
//                            neuron.weights.add(0.0);
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
                    neuronLayer.add(neuron);
                }
                
            }
            if(i!=0){
                Neuron bias = new Neuron();
                neuronLayer.add(bias);
            }
            layer.add(neuronLayer);
            System.out.println("-------");
        }
        
        //last hidden layer to output
        List<Neuron> neuronLayer = new ArrayList<Neuron>();
        for(int i = 0; i < nOutput; i++){
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
        printWeight();
    }
    
    public void printWeight(){
        for(int i = 0; i < layer.size(); i++){
            System.out.println("Layer-" + (i+1));
            for (int j=0;j<layer.get(i).size();j++) {
                System.out.println("Neuron-" + (j+1));
                for(int k = 0; k < layer.get(i).get(j).weights.size(); k++){
                    System.out.println("Weight-" + (k+1) + ": " + layer.get(i).get(j).weights.get(k));
                }
                System.out.println("-------");
            }
        }
    }
    
    public void doPerceptron(Instances data) {
//        for(int j = 0; j < layer.size(); j++){ 
//            System.out.println("Layer-" + j + " neurons: " + layer.get(j).size());
//        }
        int numAttr = data.numAttributes();
        int numInstances = data.numInstances();
        for (int epoch = 0; epoch <annOptions.maxIteration;epoch++) {
//            System.out.println("Epoch-" + (epoch+1));
            for (int i = 0; i < numInstances;i++) {
                
//                System.out.println("--------------");
//                System.out.println("Instance-" +i);
                //Hitung output neuron
                for(int j = 0; j < layer.size(); j++){ // Iterasi sebanyak jumlah layer (hidden layer + output layer)
                    for (int k = 0; k < layer.get(j).size(); k++){ // Iterasi sebanyak jumlah neuron pada layer (udah termasuk bias, bias = neuron terakhir pd layer)
                        double sum = 0.0;
                        if(j == 0){ // Untuk hidden layer pertama, value diambil dari input
                            for(int l = 0; l < numAttr; l++){                                
                                double input;
                                if(l == numAttr-1){ // bias
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
                            if(j==layer.size()-1 || (j < layer.size()-1 && k < layer.get(j).size()-1)){
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
                        }
                        
                        if (k < numAttr-1) { // if not bias
                            sum = Util.activationFunction(sum, annOptions);
                            //sigmoid
                            layer.get(j).get(k).value = sum;
                        } 
                    }
                }
                
                // let's calculate error guys
                
                int nOutputNeuron = layer.get(layer.size()-1).size();
                for (int x = 0; x< nOutputNeuron;x++) { // for output, madam
                    double output = layer.get(layer.size()-1).get(x).value;
                    double target = data.instance(i).classValue();
                    if(target == x){
                        target = 1;
                    }
                    else{
                        target = 0;
                    }
                    layer.get(layer.size()-1).get(x).error = calculateError(output,target);
                }
                boolean beforeOutput = true;
                for (int x = layer.size()-2 ; x >= 0;x--) { // for each hidden layer
                    for (int y = 0; y<layer.get(x).size();y++) { // for each neuron in hidden layer
                        double error = 0;
                        int layerSize;
                        if(beforeOutput){
                            layerSize = layer.get(x+1).size();
                        }
                        else{
                            layerSize = layer.get(x+1).size()-1;
                        }
                        for (int z = 0 ; z < layerSize;z++) { // for each neuron in next layer
                            double nextError = layer.get(x+1).get(z).error;
                            double weight = layer.get(x+1).get(z).weights.get(y);
                            error += (weight * nextError);
                        }
                        
                        double output = layer.get(x).get(y).value;
                        error *=  output * (1-output);
                        layer.get(x).get(y).error = error;
                    }
                    beforeOutput = false;
                }
                               
                //Update Weight
                for(int j = 0; j < layer.size(); j++){ // Iterasi sebanyak jumlah layer (hidden layer + output layer)
                    for (int k = 0; k < layer.get(j).size(); k++){ // Iterasi sebanyak jumlah neuron pada layer (udah termasuk bias, bias = neuron terakhir pd layer)
                        double sum = 0.0;
                        if(j == 0){ // Untuk hidden layer pertama, value diambil dari input
                            for(int l = 0; l < numAttr; l++){                                
                                double input;
                                if(l == data.numAttributes()-1){ // bias
                                    input = 1;
                                }
                                else{
                                    input = data.instance(i).value(l);                    
                                }
                                
                                double lastDeltaWeight;
                                if(epoch == 0 && i == 0){ // belum punya last Delta Weight
                                    lastDeltaWeight = 0;
                                }
                                else{
                                    lastDeltaWeight = layer.get(j).get(k).lastDeltaWeight.get(l);
                                }
                                
                                double momentum = annOptions.momentum * lastDeltaWeight;
                                
                                double deltaWeight = layer.get(j).get(k).error * input * annOptions.learningRate + momentum;
                                
                                if(epoch == 0 && i == 0){
                                    layer.get(j).get(k).lastDeltaWeight.add(deltaWeight);
                                }
                                else{
                                    layer.get(j).get(k).lastDeltaWeight.set(l,deltaWeight);
                                }
                                
                                double lastWeight = layer.get(j).get(k).weights.get(l);
                                double newWeight = lastWeight+deltaWeight;
                                layer.get(j).get(k).weights.set(l,newWeight);
                            }
                        }
                        else{ // Untuk hidden layer > 1, Value diambil dari neuron value pada layer sebelumnya
                            int nPrevLayer = layer.get(j-1).size();
                            if(j==layer.size()-1 || (j < layer.size()-1 && k < layer.get(j).size()-1)){
                                for(int l = 0; l < nPrevLayer; l++){
                                    double input;
                                    if(l == nPrevLayer-1){ // bias
                                        input = 1;
                                    }
                                    else{
                                        input = layer.get(j-1).get(l).value;
                                    }

                                    double lastDeltaWeight;
                                    if(epoch == 0 && i == 0){ // belum punya last Delta Weight
                                        lastDeltaWeight = 0;
                                    }
                                    else{
                                        lastDeltaWeight = layer.get(j).get(k).lastDeltaWeight.get(l);
                                    }
                                    double momentum = annOptions.momentum * lastDeltaWeight;

                                    double deltaWeight = layer.get(j).get(k).error * input * annOptions.learningRate + momentum;

                                    if(epoch == 0 && i == 0){
                                        layer.get(j).get(k).lastDeltaWeight.add(deltaWeight);
                                    }
                                    else{
                                        layer.get(j).get(k).lastDeltaWeight.set(l,deltaWeight);
                                    }

                                    double lastWeight = layer.get(j).get(k).weights.get(l);
                                    double newWeight = lastWeight+deltaWeight;
                                    layer.get(j).get(k).weights.set(l,newWeight);
                                }
                            }
                        }
                    }
                }
            }
            
            // error epoch
            double errorEpoch = 0;
            for (int i = 0; i < numInstances;i++) {
                for(int j = 0; j < layer.size(); j++){ // Iterasi sebanyak jumlah layer (hidden layer + output layer)
                    for (int k = 0; k < layer.get(j).size(); k++){ // Iterasi sebanyak jumlah neuron pada layer (udah termasuk bias, bias = neuron terakhir pd layer)
                        double sum = 0.0;
                        if(j == 0){ // Untuk hidden layer pertama, value diambil dari input
                            for(int l = 0; l < numAttr; l++){                                
                                double input;
                                if(l == numAttr-1){ // bias
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
                            if(j==layer.size()-1 || (j < layer.size()-1 && k < layer.get(j).size()-1)){
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
                        }

                        if (k < numAttr-1) { // if not bias
                            sum = Util.activationFunction(sum, annOptions);
                            layer.get(j).get(k).value = sum;
                        } 
                    }
                }
                int output = 0;
                if(layer.get(layer.size()-1).size() > 1){
                    for(int j = 0; j< layer.get(layer.size()-1).size(); j++){
                        if(layer.get(layer.size()-1).get(j).value > layer.get(layer.size()-1).get(output).value ){
                            output = j;
                        }
                    }
                }
                else{
                    output = (int) layer.get(layer.size()-1).get(0).value;
                }

                double target = data.instance(i).classValue();
                double tempError = target - output;
                errorEpoch+= tempError * tempError;
            }
            errorEpoch *= 0.5;
            
            if(errorEpoch < annOptions.threshold){
                break;
            }
        }
    }
    
   public int[] classifyInstances(Instances data){
        int[] classValue = new int[data.numInstances()];
        data = Util.setNominalToBinary(data);
        data = Util.setNormalize(data);
        int right = 0;
        int wrong = 0;
        
        int numInstances = data.numInstances();
        int numAttr = data.numAttributes();
        
        for (int i = 0; i < numInstances;i++) {
            for(int j = 0; j < layer.size(); j++){ // Iterasi sebanyak jumlah layer (hidden layer + output layer)
                for (int k = 0; k < layer.get(j).size(); k++){ // Iterasi sebanyak jumlah neuron pada layer (udah termasuk bias, bias = neuron terakhir pd layer)
                    double sum = 0.0;
                    if(j == 0){ // Untuk hidden layer pertama, value diambil dari input
                        for(int l = 0; l < numAttr; l++){                                
                            double input;
                            if(l == numAttr-1){ // bias
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
                        if(j==layer.size()-1 || (j < layer.size()-1 && k < layer.get(j).size()-1)){
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
                    }

                    if (k < numAttr-1) { // if not bias
                        sum = Util.activationFunction(sum, annOptions);
                        layer.get(j).get(k).value = sum;
                    } 
                }
            }
            
            if(layer.get(layer.size()-1).size() > 1){
                for(int j = 0; j< layer.get(layer.size()-1).size(); j++){
                    if(layer.get(layer.size()-1).get(j).value > layer.get(layer.size()-1).get(classValue[i]).value ){
                        classValue[i] = j;
                    }
                }
            }
            else{
                classValue[i] = (int) layer.get(layer.size()-1).get(0).value;
            }
            
            double target = data.instance(i).classValue();
            double output = classValue[i];
            
            System.out.println("Intance-" + i + " target: " + target + " output: " + output);
            if(target == output){
                right = right + 1;
            }
        }
        System.out.println("Percentage: " + ((double)right/(double)data.numInstances()));
        printWeight();
        return classValue;
   }
    
    public double calculateError(double output, double target){
        return (output *(1-output) *(target-output));
    }
    
    public double calculateError(double output, double errorAfter, double weight) {
        return (output * (1 - output) * errorAfter * weight);
    }
}
