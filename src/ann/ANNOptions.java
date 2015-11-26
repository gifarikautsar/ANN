package ann;


import java.util.ArrayList;
import java.util.List;
import weka.core.Instances;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author gifarikautsar
 */
public class ANNOptions {
    public int weightOpt, topologyOpt, activationFunctionOpt, maxIteration, hiddenLayer;
    public double learningRate, threshold, momentum;
    public List<Integer> layerNeuron;
    public Instances dataset;
    
    public ANNOptions() {
        layerNeuron = new ArrayList<Integer>();
    }
}