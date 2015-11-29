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
    
    public ANNOptions() {
        layerNeuron = new ArrayList<Integer>();
        weightOpt = 1;
        topologyOpt = 4;
        activationFunctionOpt = 1;
        hiddenLayer = 1;
        layerNeuron.add(2);
        maxIteration = 500;
        momentum = 0.2;
        learningRate = 0.3;
        threshold = 0.2;
    }
}
