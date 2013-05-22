package org.mephi.addClasses;

public class MyMath {
	
	public static double[] getSubArr(int startIndex, int leng, double[] sourceArr)
	{
		double[] res = new double[leng];
		
		for(int i = 0; i < leng; ++i)
		{
			res[i] = sourceArr[i + startIndex];
		}
		
		return res;
	}
	
	public static double[] replaceFromArray(int index, double[] sourceArr)
	{
		double[] res = new double[sourceArr.length - 1];
		
		for(int i = 0; i < res.length; ++i)
		{
			if(i < index)
			{
				res[i] = sourceArr[i];
			}
			else
			{
				res[i] = sourceArr[i + 1];
			}
		}
		
		return res;
	}
	
	public static double countDev(double[] arr1,double[] arr2) throws Exception
	{
		if(arr1.length == arr2.length)
		{
			double dev = 0.0;
			for(int i = 0 ; i < arr1.length; ++i)
			{
				dev += Math.abs(arr1[i] - arr2[i]);
			}
			
			return dev;
		}
		else
		{
			throw new Exception("Wrong arr size");
		}
	}
	
	public static int minInArr(double[] arr)
	{
		double min = arr[0];
		int n = 0;
		
		for(int i = 0; i < arr.length; ++i)
		{
			if(min > arr[i])
			{
				min = arr[i];
				n = i;
			}
		}
		
		return n;
	}
	
	public static int maxInArr(double[] arr)
	{
		double max = arr[0];
		int n = 0;
		
		for(int i = 0; i < arr.length; ++i)
		{
			if(max < arr[i])
			{
				max = arr[i];
				n = i;
			}
		}
		
		return n;
	}
	
	public static int secMinInArr(double[] arr)
	{
		double min = arr[0];
		int n = 0;
		int s = 0;
		
		for(int i = 0; i < arr.length; ++i)
		{
			if(min > arr[i])
			{
				min = arr[i];
				s = n;
				n = i;
			}
		}
		
		return s;
	}
}
