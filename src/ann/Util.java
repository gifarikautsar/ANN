/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import static java.lang.Math.exp;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

/**
 *
 * @author Tony
 */
public class Util {
    public static Instances setNominalToBinary(Instances instances) {
        NominalToBinary ntb = new NominalToBinary();
        Instances newInstances = null;
        try {
            ntb.setInputFormat(instances);
            newInstances = new Instances(Filter.useFilter(instances, ntb));
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        return newInstances;
    }
    
    public static double activationFunction(double output, ANNOptions annOptions){
        double activateValue = output;
        if(annOptions.topologyOpt == 1){ // Perceptron Training Rule
            if(annOptions.activationFunctionOpt == 1){ // Step
                if(output < 0){
                    activateValue = 0;
                }
                else{
                    activateValue = 1;
                }
            }
            else if(annOptions.activationFunctionOpt == 2){ // Sign
                if(output < 0){
                    activateValue = -1;
                }
                else{
                    activateValue = 1;
                }
            }
            else{ // Sigmoid
                activateValue = 1/(1+exp(-output));
            }
        }
        return activateValue;
    }
}
