package assignment2;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksOpt {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    
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
		int[] iters = {10, 100, 500, 1000, 2000, 3000, 4000, 5000};
		int n_rounds = 3;
		double start_time = 0;
		double end_time = 0;
		
		ArrayList<ArrayList<ArrayList<Double>>> results = new ArrayList<ArrayList<ArrayList<Double>>>();
		ArrayList<ArrayList<ArrayList<Double>>> time = new ArrayList<ArrayList<ArrayList<Double>>>();
		
		for (int iter : iters) {
			ArrayList<ArrayList<Double>> results_iter = new ArrayList<ArrayList<Double>>();
			ArrayList<ArrayList<Double>> time_iter = new ArrayList<ArrayList<Double>>();
			
			for (int k=0; k<n_rounds; k++) {
				ArrayList<Double> results_round = new ArrayList<Double>();
				ArrayList<Double> time_round = new ArrayList<Double>();
    	
		        int[] ranges = new int[N];
		        Arrays.fill(ranges, 2);
		        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
		        Distribution odd = new DiscreteUniformDistribution(ranges);
		        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
		        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
		        CrossoverFunction cf = new SingleCrossOver();
		        Distribution df = new DiscreteDependencyTree(.1, ranges); 
		        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
		        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
		        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
		        
		        start_time = System.nanoTime();
		        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
		        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
		        fit.train();
		        end_time = System.nanoTime();
		        results_round.add(ef.value(rhc.getOptimal()));
		        time_round.add(end_time-start_time);
		        //System.out.println("RHC: " + ef.value(rhc.getOptimal()));
		        
		        start_time = System.nanoTime();
		        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
		        fit = new FixedIterationTrainer(sa, iter);
		        fit.train();
		        end_time = System.nanoTime();
		        results_round.add(ef.value(sa.getOptimal()));
		        time_round.add(end_time-start_time);
		        //System.out.println("SA: " + ef.value(sa.getOptimal()));
		        
		        start_time = System.nanoTime();
		        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
		        fit = new FixedIterationTrainer(ga, iter);
		        fit.train();
		        end_time = System.nanoTime();
		        results_round.add(ef.value(ga.getOptimal()));
		        time_round.add(end_time-start_time);
		        //System.out.println("GA: " + ef.value(ga.getOptimal()));
		        
		        start_time = System.nanoTime();
		        MIMIC mimic = new MIMIC(200, 20, pop);
		        fit = new FixedIterationTrainer(mimic, iter);
		        fit.train();
		        end_time = System.nanoTime();
		        results_round.add(ef.value(mimic.getOptimal()));
		        time_round.add(end_time-start_time);
		        //System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
		        
		        results_iter.add(results_round);
		        time_iter.add(time_round);
			}
	        System.out.println(results_iter);
	        results.add(results_iter);
	        time.add(time_iter);
		}
		System.out.println(results);
		write_output_to_file("/Users/yuezha01/projects/ml7641_2018/", "results_4peaks.txt", results.toString(), true);
		write_output_to_file("/Users/yuezha01/projects/ml7641_2018/", "time_4pearks.txt", time.toString(), true);
    }
}
