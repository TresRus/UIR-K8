package org.mephi.predict;

import org.mephi.neuralNet.*;

public class Predict {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try{
			//Neuron n = new Neuron(1,2,0.5);
			//Layer l = new Layer(1,2,5);
			NeuroNet net = new NeuroNet(new int[] {4,3,1}, new int[] {1,2,3},2);
			double[] in = new double[] {20,-16};
			double[] test = new double[] {3};
			for (int i = 0; i < 400; ++i)
			{
				double[] out = net.trainNet(in, test, 0.2);
				System.out.printf("%d:\n",i);
				for (double o : out)
				{
					System.out.println(o);
				}
			}
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
	}

}
