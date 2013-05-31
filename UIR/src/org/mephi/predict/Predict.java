package org.mephi.predict;

import java.util.Random;

import org.mephi.addClasses.MyMath;
import org.mephi.neuralNet.*;

public class Predict {

	private static Random gen = new Random();
	
	public static void main(String[] args) {
		try {
			DataIO data = new DataIO();
			NeuroNet net;
			Train tga;
			Train tbe;
			FilterKalman filter;
			
			double[] dataIn;
			double[] filtDataIn;
			double[] allData;
			double[] filtAllData;
			
			double[] d1;
			double[] delta1;
			double[] filtDelta1;
			
			double[] d2;
			double[] delta2;
			double[] filtDelta2;
			
			double maxValue;
			
			double maxValue1;
			double maxValue2;
			
			double minValue1;
			double minValue2;
			
			double alpha = 0.6;
			
			data.readData("E:\\data.xls");
			
			for(int i = 0; i < data.dataList.size(); ++i) {
				System.out.printf("Run for data #%d :\n", i);
				
				if(i == 6) {
					data.writeTestRun("E:\\netRes.xls", i);
				}
			}
			
			data.writeDate("E:\\filterRes.xls");
			
			System.out.println("end");
			
		} catch(Exception e) {
			System.out.println(e.getMessage());
		}
	}
}
