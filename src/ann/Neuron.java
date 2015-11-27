/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author gifarikautsar
 */
public class Neuron {
    public List<Double> weights;
    public double value;
    
    public Neuron(){
        weights = new ArrayList<Double>();
    }
}
