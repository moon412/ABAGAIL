package assignment2;
import java.util.Arrays;
import java.util.Date;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

public class test {
	public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }
    
	public static void main(String[] args) {
		//int[] a = new int[3];
		/**
		double start_time = 0;
		double end_time = 0;
		start_time = System.nanoTime();
		ArrayList<ArrayList<Double>> b = new ArrayList<ArrayList<Double>>(3);
		for (int j=0; j<3; j++) {
			ArrayList<Double> a = new ArrayList<Double>(5);
			for (int i=0; i<5; i++) {
				a.add(j+0.1*i);
			}
			b.add(a);
		}
		System.out.println(b.toString());
		int[][] c = new int[2][3];
		for (int i=0; i<2; i++) {
			for (int j=0; j<3; j++) {
				c[i][j] = i+j;
			}
		}
		end_time = System.nanoTime();
		System.out.println(c[0][0]);
		System.out.println("Time: " + Double.toString(end_time-start_time));
		write_output_to_file("/Users/yuezha01/projects/ml7641_2018/", "c_test.csv", c.toString(), true);
		*/
		double[] guess = new double[26];
		for (double x : guess)
			System.out.println(x);
	}
}

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
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 16, hiddenLayer = 8, outputLayer = 26, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();
    
    // this dataset is used to train the network
    private static DataSet set = new DataSet(instances);
    
    // a list of transfer function
    //private static HyperbolicTangentSigmoid transfer = new HyperbolicTangentSigmoid();
    private static DifferentiableActivationFunction[] transfers = new DifferentiableActivationFunction[2];
    
    private static BackPropagationNetwork network = new BackPropagationNetwork();
    
    // Create one optimization problem given a network and measure
    private static NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
    		
    // Create one optimization algorithm
    private static OptimizationAlgorithm oa = new RandomizedHillClimbing(nnop);
    
    private static String results = "";
    
    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        transfers[0] = new LogisticSigmoid();
        transfers[1] = new HyperbolicTangentSigmoid();
        network = factory.createClassificationNetwork(
        		new int[] {inputLayer, hiddenLayer, outputLayer}, transfers);
        nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
        oa = new RandomizedHillClimbing(nnop);
        
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
        for(int j = 0; j < instances.length; j++) {
        		// input data into network
            network.setInputValues(instances[j].getData());
            // run the network on the input
            network.run();
            
            // get actual label from instance[j]
            Instance actual = instances[j].getLabel();
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
        correct = 20000 - incorrect;
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
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();
                
                // two instances: output and example
                Instance output = instances[j].getLabel();
                Instance example = new Instance(network.getOutputValues());
                //System.out.println("output:"+output);     
                example.setLabel(new Instance(network.getOutputValues()));
                //System.out.println("example"+example);
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

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

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // use the actual label
            instances[i].setLabel(new Instance(attributes[i][1]));
        }

        return instances;
    }
}

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
    private static Instance[] instances = initializeInstances();
    private static Instance[] training = new Instance[10000];
    private static Instance[] test = new Instance[10000];

    private static int inputLayer = 16, hiddenLayer = 8, outputLayer = 26, trainingIterations = 2;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();
    
    // this dataset is used to train the network
    private static DataSet set = new DataSet(instances);
    
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
    		for(int c=0; c<10000; c++) {
    			training[c] = instances[c];
    			test[c] = instances[c+10000];
    		}
    		System.out.println(training[0]);
    		System.out.println(test[0]);
    	
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
        for(int j = 0; j < instances.length; j++) {
        		// input data into network
            network.setInputValues(instances[j].getData());
            // run the network on the input
            network.run();
            
            // get actual label from instance[j]
            Instance actual = instances[j].getLabel();
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
        correct = 20000 - incorrect;
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
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();
                
                // two instances: output and example
                Instance output = instances[j].getLabel();
                Instance example = new Instance(network.getOutputValues());
                //System.out.println("output:"+output);     
                example.setLabel(new Instance(network.getOutputValues()));
                //System.out.println("example"+example);
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

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

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // use the actual label
            instances[i].setLabel(new Instance(attributes[i][1]));
        }

        return instances;
    }
}


