/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author gifarikautsar
 */
public class Neuron implements Serializable{
    public List<Double> weights; // all previous weight
    public double error; // next error
    public double value;
    public List<Double> lastDeltaWeight;
    
    public Neuron(){
        weights = new ArrayList<Double>();
        lastDeltaWeight = new ArrayList<Double>();
    }
}
