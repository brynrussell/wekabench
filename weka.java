import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import java.text.DecimalFormat;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.classifiers.Classifier;



public class weka {

	public static void main(String[] args) throws Exception{



		/////////////LOADING THE DATA

		// Read in the training set
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("/media/bryn/DATADRIVE1/cancer/data/170119 - merged tables ready for weka/subset/subset.arff"));
		// Form the training set instance from the reader
		Instances initialtrain = new Instances (breader);
		initialtrain.setClassIndex(initialtrain.numAttributes() -1);




		//create the classifiers 
		Classifier theclassifier = new J48();
		//		cvj48.setUnpruned(false);




		int perc=20;
		int startperc = perc;
		int percincrement = 1;
		int numofresults = (int) (1+Math.floor((100-perc)/percincrement));
		int count = 0;

		double [][] crossvalresults = new double[numofresults][3];
		double [][] realresults = new double[numofresults][2];




		System.out.println("Starting cross validation and model evaluation using "+perc+" % of the data");
		while(perc<=100)
		{

			int numtest=1000; //decide a number of times to do the cross validation 
			double[] cvpctcorrectarry= new double[numtest];
			double[] pctcorrectarry= new double[numtest];
			for(int i=0;i<numtest;i++)
			{

				initialtrain.randomize(new Random(0)); //Randomise list order
				int trainsize = initialtrain.numInstances() * perc/100; //get the size of the training/test set
				int testsize = initialtrain.numInstances() - trainsize; 
				Random rannnum = new Random(0);
				Instances train = new Instances(initialtrain,0,trainsize);   //create the training/test set
				Instances test = new Instances(initialtrain,trainsize,testsize);

				Evaluation cvevalj48 = new Evaluation(train);
				cvevalj48.crossValidateModel(theclassifier, train, 10, rannnum); // Do the cross validation
				cvpctcorrectarry[i]=cvevalj48.pctCorrect();

				// Evaluate the model using the test set and get the percentage of correct results
				Classifier realmodel = new J48();
				realmodel.buildClassifier(train);
				Evaluation eval = new Evaluation(train);
				eval.evaluateModel(realmodel, test, args);
				pctcorrectarry[i]=eval.pctCorrect();

			}

			//work out the average pct correct from the repeated cross validations
			double cvpctcorrectsum=0;
			for(int i=0;i<cvpctcorrectarry.length; i++)
			{
				cvpctcorrectsum=cvpctcorrectsum+cvpctcorrectarry[i];
			}
			double cvpctcorrectav=cvpctcorrectsum/cvpctcorrectarry.length;


			//Work out the standard deviation of the pct correct
			double cvstdv=0;
			for(int i=0; i<cvpctcorrectarry.length; i++)
			{
				cvstdv=cvstdv+Math.pow((cvpctcorrectav-cvpctcorrectarry[i]),2);
				


			}
			cvstdv=cvstdv/cvpctcorrectarry.length;
			cvstdv=Math.sqrt(cvstdv);


			//work out the average pct correct from the actual results
			double pctcorrectsum=0;
			for(int i=0;i<cvpctcorrectarry.length; i++)
			{
				pctcorrectsum=pctcorrectsum+pctcorrectarry[i];
			}
			double pctcorrectav=pctcorrectsum/pctcorrectarry.length;


			crossvalresults[count][0]=perc;
			crossvalresults[count][1]=cvpctcorrectav;
			crossvalresults[count][2]=cvstdv;


			realresults[count][1]=pctcorrectav;
			realresults[count][0]=perc;








			count++;
			perc=perc+percincrement;


			//Progress counter
			if(perc<=100)
			{
				System.out.println("A cycle of "+numtest+" reps is done... Incrementing the percetage split to  "+(perc)+" %");
			}
			else{System.out.println("Finished incrementing the percentage split!");}




		}         // 



		//Print the results from Cross validation and the remaining test data
		System.out.println("Done\n\n\n---------------");
		System.out.println("Restuls:\n---------------");
		System.out.println("% of data for training     |     % correct from CV     |     % correct from test");
		for(int i=0; i<crossvalresults.length; i++)
		{
			System.out.println(new DecimalFormat("#.##").format(crossvalresults[i][0])+"                               "+new DecimalFormat("#.#").format(crossvalresults[i][1])+" +- "+new DecimalFormat("#.##").format(crossvalresults[i][2])+"                         "+new DecimalFormat("#.#").format(realresults[i][1]));
		}


		//Print to file
		try{
		    PrintWriter writer = new PrintWriter("Output.txt", "UTF-8");
		    writer.println("#Stated using "+startperc+"% of the data incrementing an extra "+percincrement+"% each run");
		    for(int i=0; i<crossvalresults.length; i++)
			{
		    	writer.println(new DecimalFormat("#.##").format(crossvalresults[i][0])+"		"+new DecimalFormat("#.#").format(crossvalresults[i][1])+"		"+new DecimalFormat("#.##").format(crossvalresults[i][2]));
			}
		    
		    
		    
		    
		    
		    
		    writer.close();
		} catch (IOException e) {
		   // do something
		}

	}



}


