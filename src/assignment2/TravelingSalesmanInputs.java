package assignment2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Date;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
*
* @author Andrew Guillory gtg008g@mail.gatech.edu
* @version 1.0
*/
public class TravelingSalesmanInputs {
    private static final int iter = 1000; 
    /**
     * The test main
     * @param args ignored
     */
    
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
    		int[] inputs = {10, 30, 50, 70, 90, 110};
    		int n_rounds = 3;
    		double start_time = 0;
    		double end_time = 0;
    		
    		ArrayList<ArrayList<ArrayList<Double>>> results = new ArrayList<ArrayList<ArrayList<Double>>>();
    		ArrayList<ArrayList<ArrayList<Double>>> time = new ArrayList<ArrayList<ArrayList<Double>>>();
    		
    		for (int input : inputs) {
    			ArrayList<ArrayList<Double>> results_iter = new ArrayList<ArrayList<Double>>();
    			ArrayList<ArrayList<Double>> time_iter = new ArrayList<ArrayList<Double>>();
    			
    			for (int k=0; k<n_rounds; k++) {
    				ArrayList<Double> results_round = new ArrayList<Double>();
    				ArrayList<Double> time_round = new ArrayList<Double>();
    				
		        Random random = new Random();
		        // create the random points
		        double[][] points = new double[input][2];
		        for (int i = 0; i < points.length; i++) {
		            points[i][0] = random.nextDouble();
		            points[i][1] = random.nextDouble();   
		        }
		        // for rhc, sa, and ga we use a permutation based encoding
		        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
		        Distribution odd = new DiscretePermutationDistribution(input);
		        NeighborFunction nf = new SwapNeighbor();
		        MutationFunction mf = new SwapMutation();
		        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
		        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
		        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
		        
		        start_time = System.nanoTime();
		        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
		        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
		        fit.train();
		        end_time = System.nanoTime();
		        results_round.add(ef.value(rhc.getOptimal()));
		        time_round.add(end_time-start_time);
		        //System.out.println(ef.value(rhc.getOptimal()));
		        
		        start_time = System.nanoTime();
		        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
		        fit = new FixedIterationTrainer(sa, iter);
		        fit.train();
		        end_time = System.nanoTime();
		        results_round.add(ef.value(sa.getOptimal()));
		        time_round.add(end_time-start_time);
		        //System.out.println(ef.value(sa.getOptimal()));
		        
		        start_time = System.nanoTime();
		        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
		        fit = new FixedIterationTrainer(ga, iter);
		        fit.train();
		        end_time = System.nanoTime();
		        results_round.add(ef.value(ga.getOptimal()));
		        time_round.add(end_time-start_time);
		        //System.out.println(ef.value(ga.getOptimal()));
		        
		        // for mimic we use a sort encoding
		        ef = new TravelingSalesmanSortEvaluationFunction(points);
		        int[] ranges = new int[input];
		        Arrays.fill(ranges, input);
		        odd = new  DiscreteUniformDistribution(ranges);
		        Distribution df = new DiscreteDependencyTree(.1, ranges); 
		        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
		        
		        start_time = System.nanoTime();
		        MIMIC mimic = new MIMIC(200, 100, pop);
		        fit = new FixedIterationTrainer(mimic, iter);
		        fit.train();
		        end_time = System.nanoTime();
		        results_round.add(ef.value(mimic.getOptimal()));
		        time_round.add(end_time-start_time);
		        //System.out.println(ef.value(mimic.getOptimal()));
		        
		        results_iter.add(results_round);
		        time_iter.add(time_round);
    			}
	        System.out.println(results_iter);
	        results.add(results_iter);
	        time.add(time_iter);
    		}
    		System.out.println(results);
    		write_output_to_file("/Users/yuezha01/projects/ml7641_2018/", "results_tsp_inputs.txt", results.toString(), true);
    		write_output_to_file("/Users/yuezha01/projects/ml7641_2018/", "time_tsp_inputs.txt", time.toString(), true);
    }
}
