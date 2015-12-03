/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.io.Serializable;
import static java.lang.Math.exp;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Tony
 */
public class Util implements Serializable {  
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
        if(annOptions.topologyOpt == 4){
            activateValue = 1/(1+exp(-output));
        }
        return activateValue;
    }
}
