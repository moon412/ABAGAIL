package assignment2;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import func.nn.activation.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AbaloneOpt {
    private static Instance[] training = initializeTrain();
    private static Instance[] test = initializeTest();

    private static int inputLayer = 16, hiddenLayer = 8, outputLayer = 26, trainingIterations = 100;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();
    
    // this dataset is used to train the network
    private static DataSet set = new DataSet(training);
    
    // a list of transfer function
    private static HyperbolicTangentSigmoid transfer = new HyperbolicTangentSigmoid();
    //private static DifferentiableActivationFunction[] transfers = new DifferentiableActivationFunction[2];
    //transfers[0] = new LogisticSigmoid();
    //transfers[1] = new HyperbolicTangentSigmoid();
    
    private static BackPropagationNetwork network = factory.createClassificationNetwork(
    		new int[] {inputLayer, hiddenLayer, outputLayer}, transfer);
    
    // Create one optimization problem given a network and measure
    private static NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
    		
    // Create one optimization algorithm
    private static OptimizationAlgorithm oa = new RandomizedHillClimbing(nnop);
    //private static OptimizationAlgorithm oa = new StandardGeneticAlgorithm(200, 100, 10, nnop);
    
    private static String results = "";
    
    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
    		System.out.println(training.length);
    		System.out.println(test.length);
    	
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        train(oa, network, "RHC"); //trainer.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        // store optimal weights from oa into an instance
        Instance optimalInstance = oa.getOptimal();
        // set network's weights to the weights in the instance
        network.setWeights(optimalInstance.getData());

        start = System.nanoTime();
        // go through the sample one by one as an input
        for(int j = 0; j < test.length; j++) {
        		// input data into network
            network.setInputValues(test[j].getData());
            // run the network on the input
            network.run();
            
            // get actual label from instance[j]
            Instance actual = test[j].getLabel();
            // get predicted label from network for this input
            double[] temp = new double[26];
            temp[network.getOutputValues().argMax()] = 1;
            Instance predicted = new Instance(temp);
            predicted.setLabel(new Instance(temp));
            
            //System.out.println("Actual:" + actual);
            //System.out.println("Predicted:" + predicted);
            //System.out.println(measure.value(actual, predicted));
            incorrect += measure.value(actual, predicted);
            // test error but this is on the same training data
            //double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        correct = 10000 - incorrect;
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        results +=  "\nResults for " + "RHC" + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        
        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
        	    // train on all the data
            oa.train();

            double error = 0; // training error
            for(int j = 0; j < training.length; j++) {
                network.setInputValues(training[j].getData());
                network.run();
                
                // two instances: output and example
                Instance output = training[j].getLabel();
                Instance example = new Instance(network.getOutputValues());
                //System.out.println("output:"+output);     
                example.setLabel(new Instance(network.getOutputValues()));
                //System.out.println("example"+example);
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeTrain() {

        double[][][] attributes = new double[20000][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("/Users/yuezha01/projects/ml7641_2018/assignment_2/letter_onehot.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[16]; // 16 attributes
                attributes[i][1] = new double[26]; // 26 labels

                for(int j = 0; j < 16; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                for(int k = 0; k < 26; k++)
                		attributes[i][1][k] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length/2];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // use the actual label
            instances[i].setLabel(new Instance(attributes[i][1]));
        }

        return instances;
    }
    
    private static Instance[] initializeTest() {

        double[][][] attributes = new double[20000][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("/Users/yuezha01/projects/ml7641_2018/assignment_2/letter_onehot.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[16]; // 16 attributes
                attributes[i][1] = new double[26]; // 26 labels

                for(int j = 0; j < 16; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                for(int k = 0; k < 26; k++)
                		attributes[i][1][k] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length/2];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i+10000][0]);
            // use the actual label
            instances[i].setLabel(new Instance(attributes[i+10000][1]));
        }

        return instances;
    }
}

