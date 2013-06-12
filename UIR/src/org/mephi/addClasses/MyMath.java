package org.mephi.addClasses;

public class MyMath {
	
	/**
	 * Get sub array form source array.
	 * @param startIndex = first index of sub array
	 * @param leng
	 * @param sourceArr
	 * @return
	 */
	public static double[] getSubArr(int startIndex, int leng, double[] sourceArr) {
		double[] res = new double[leng];
		
		for(int i = 0; i < leng; ++i) {
			res[i] = sourceArr[i + startIndex];
		}
		
		return res;
	}
	
	public static double[] replaceFromArray(int index, double[] sourceArr) {
		double[] res = new double[sourceArr.length - 1];
		
		for(int i = 0; i < res.length; ++i) {
			if(i < index) {
				res[i] = sourceArr[i];
			} else {
				res[i] = sourceArr[i + 1];
			}
		}
		
		return res;
	}
	
	public static double[] appendToArr(double[] arr, double val) {
		double[] res = new double[arr.length+1];
		for(int i = 0; i< arr.length; ++i) {
			res[i] = arr[i];
		}
		res[arr.length] = val;
		return res;
	}
	
	public static double countDev(double[] arr1,double[] arr2) throws Exception {
		if(arr1.length == arr2.length) {
			double dev = 0.0;
			for(int i = 0 ; i < arr1.length; ++i) {
				dev += Math.abs(arr1[i] - arr2[i]);
			}
			
			return dev;
		} else {
			throw new Exception("Wrong arr size");
		}
	}
	
	public static int minInArr(double[] arr) {
		double min = arr[0];
		int n = 0;
		
		for(int i = 0; i < arr.length; ++i) {
			if(min > arr[i]) {
				min = arr[i];
				n = i;
			}
		}
		
		return n;
	}
	
	public static double getMinInArr(double[] arr) {
		double min = arr[0];
		
		for(int i = 0; i < arr.length; ++i) {
			if(min > arr[i]) {
				min = arr[i];
			}
		}
		
		return min;
	}
	
	public static int maxInArr(double[] arr) {
		double max = arr[0];
		int n = 0;
		
		for(int i = 0; i < arr.length; ++i) {
			if(max < arr[i]) {
				max = arr[i];
				n = i;
			}
		}
		
		return n;
	}
	
	public static double getMaxInArr(double[] arr) {
		double max = arr[0];
		
		for(int i = 0; i < arr.length; ++i) {
			if(max < arr[i]) {
				max = arr[i];
			}
		}
		
		return max;
	}
	
	public static int secMinInArr(double[] arr) {
		double min = arr[0];
		int n = 0;
		int s = 0;
		
		for(int i = 0; i < arr.length; ++i) {
			if(min > arr[i]) {
				min = arr[i];
				s = n;
				n = i;
			}
		}
		
		return s;
	}
	
	public static double[] balanceData(double[] data, double alpha, double n) {
		double[] res = new double[data.length];
		double[] balanced = new double[data.length];
		
		double averVal = 0.0;
		double msd = 0.0;
		double lambda = 0.0;
		
		for(int i = 0; i < data.length; ++i) {
			if(i == 0) {
				balanced[i] = (data[i] + data[i + 1] + data[i + 2])/3;
			} else {
				balanced[i] = data[i] * alpha + (1 - alpha) * balanced[i-1];
			}
			averVal += data[i];
		}
		averVal /= data.length;
		
		for(int i = 0; i < data.length; ++i) {
			msd += Math.pow((data[i] - averVal), 2);
		}
		msd /= data.length - 1;
		msd = Math.sqrt(msd);
		
		res[0] = data[0];
		for(int i = 1; i < data.length; ++i) {
			lambda = Math.abs(data[i] - res[i-1])/msd;
			if(lambda > n) {
				res[i] = balanced[i];
			} else {
				res[i] = data[i];
			}
		}
		
		return res;
	}
	
	public static double[] minusArr(double[] a1, double[] a2) {
		double[] res = new double[a1.length];
		for(int i = 0; i < a1.length; ++i) {
			res[i] = a1[i] - a2[i];
		}
		return res;
	}
}
