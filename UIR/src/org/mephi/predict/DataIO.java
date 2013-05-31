package org.mephi.predict;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.List;

import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.mephi.addClasses.MyMath;
import org.mephi.neuralNet.NeuroNet;
import org.mephi.neuralNet.Train;
import org.mephi.neuralNet.TrainGA;

public class DataIO {
	private List<Date[]> dateVal;
	private List<String> names;
	public List<double[]> dataList;
	public List<double[]> filtred1;
	public List<double[]> delta1;
	public List<double[]> filtred2;
	public List<double[]> delta2;
	public List<double[]> filtred3;
	public List<double[]> delta3;
	public List<double[]> filtred4;
	
	public DataIO() {
		dateVal = new ArrayList<Date[]>();
		names = new ArrayList<String>();
		dataList = new ArrayList<double[]>();
		filtred1 = new ArrayList<double[]>();
		delta1 = new ArrayList<double[]>();
		filtred2 = new ArrayList<double[]>();
		delta2 = new ArrayList<double[]>();
		filtred3 = new ArrayList<double[]>();
		delta3 = new ArrayList<double[]>();
		filtred4 = new ArrayList<double[]>();
	}
	
	public String getName(int num) {
		return names.get(num);
	}
	
	public void readData(String fileName) throws IOException {
		dateVal.clear();
		names.clear();
		dataList.clear();
		filtred1.clear();
		delta1.clear();
		filtred2.clear();
		delta2.clear();
		filtred3.clear();
		delta3.clear();
		filtred4.clear();

		List<Double> ad = new ArrayList<Double>();
		List<Date> dt = new ArrayList<Date>();
		
		FileInputStream file = new FileInputStream(new File(fileName));
		HSSFWorkbook workbook = new HSSFWorkbook(file);
		HSSFSheet sheet = workbook.getSheetAt(1);
		
		Iterator<Row> rowIterator = sheet.iterator();
		Row row = rowIterator.next();
		
		double prevATMnum = 0.0;
		double summ;
		Date d;
		
		while(rowIterator.hasNext()) {
			row = rowIterator.next();
			
			Cell ATMcell = row.getCell(0);
			Cell dateCell = row.getCell(1);
			Cell summCell = row.getCell(4);
			
			summ = summCell.getNumericCellValue();
			d = dateCell.getDateCellValue();
			
			if(prevATMnum == ATMcell.getNumericCellValue()) {
				ad.add(summ);
				dt.add(d);
			} else {
				processLists(ad,dt,prevATMnum);
				
				ad = new ArrayList<Double>();
				dt = new ArrayList<Date>();
				
				ad.add(summ);
				dt.add(d);
			}
			prevATMnum = ATMcell.getNumericCellValue();
		}
		
		processLists(ad,dt,prevATMnum);
		
		file.close();
	}
	
	private void processLists(List<Double> ad, List<Date> dt, double prevATMnum) {
		FilterKalman filter;
		
		Date[] date;
		double[] atmdata;
		double[] balanc1;
		double[] filter1;
		double[] d1;
		double[] balanc2;
		double[] filter2;
		double[] d2;
		double[] balanc3;
		double[] filter3;
		double[] d3;
		double[] filter4;
		
		double maxValue;
		
		if(ad.size() > 0) {
			atmdata = new double[ad.size()];
			date = new Date[ad.size()];
			
			for(int i = 0; i < ad.size(); ++i) {
				atmdata[i] = ad.get(i).doubleValue();
				date[i] = dt.get(i);
			}
			
			dataList.add(atmdata);
			dateVal.add(date);
			
			balanc1 = MyMath.balanceData(atmdata,0.7,0.8);
			
			maxValue = MyMath.getMaxInArr(atmdata) * 1.3;
			
			filter = new FilterKalman(maxValue * 0.0007, maxValue * 0.40);
			filter1 = filter.runFilter(balanc1);
			filter1 = MyMath.balanceData(filter1,0.5,0.3);
			filtred1.add(filter1);
			
			d1 = new double[filter1.length];
			
			for(int i = 0; i < filter1.length; ++i) {
				d1[i] = atmdata[i] - filter1[i];
			}
			delta1.add(d1);
			
			balanc2 = MyMath.balanceData(d1,0.7,1.0);
			
			maxValue = MyMath.getMaxInArr(d1) * 1.3;
			
			filter = new FilterKalman(maxValue * 0.15, maxValue * 0.35);
			filter2 = filter.runFilter(balanc2);
			filter2 = MyMath.balanceData(filter2,0.7,0.5);
			filtred2.add(filter2);
			
			d2 = new double[filter2.length];
			
			for(int i = 0; i < filter2.length; ++i) {
				d2[i] = d1[i] - filter2[i];
			}
			delta2.add(d2);
			
			balanc3 = MyMath.balanceData(d2,0.7,1.0);
			
			maxValue = MyMath.getMaxInArr(d2) * 1.3;
			
			filter = new FilterKalman(maxValue * 0.35, maxValue * 0.05);
			filter3 = filter.runFilter(balanc3);
			filter3 = MyMath.balanceData(filter3,0.8,0.8);
			filtred3.add(filter3);
			
			d3 = new double[filter3.length];
			
			for(int i = 0; i < filter3.length; ++i) {
				d3[i] = d2[i] - filter3[i];
			}
			delta3.add(d3);
			
			filter4 = MyMath.balanceData(d3,0.6,1.0);
			filtred4.add(filter4);
			
			names.add(String.format("%.0f", prevATMnum));
		}
	}
	
	public void writeDate(String fileName) throws IOException {
		DateFormat df = DateFormat.getDateInstance(DateFormat.SHORT);
		
		HSSFWorkbook workbook = new HSSFWorkbook();
		
		for(int i = 0; i < names.size(); ++i) {
			
			HSSFSheet sheet = workbook.createSheet(names.get(i));
			
			Row row;
			Cell cell;
			
			row = sheet.createRow(0);
			
			cell = row.createCell(0);
			cell.setCellValue("Date");
			
			cell = row.createCell(1);
			cell.setCellValue("Data");
			
			cell = row.createCell(2);
			cell.setCellValue("Filtred 1");
			
			cell = row.createCell(3);
			cell.setCellValue("Delta 1");

			cell = row.createCell(4);
			cell.setCellValue("Filtred 2");
			
			cell = row.createCell(5);
			cell.setCellValue("Delta 2");
			
			cell = row.createCell(6);
			cell.setCellValue("Filtred 3");
			
			cell = row.createCell(7);
			cell.setCellValue("Delta 3");
			
			cell = row.createCell(8);
			cell.setCellValue("Filtred 4");
			
			for(int j = 0; j < dateVal.get(i).length; ++j) {
				row = sheet.createRow(j+1);
				
				cell = row.createCell(0);
				cell.setCellValue(df.format(dateVal.get(i)[j]));
				
				cell = row.createCell(1);
				cell.setCellValue(dataList.get(i)[j]);
				
				cell = row.createCell(2);
				cell.setCellValue(filtred1.get(i)[j]);
				
				cell = row.createCell(3);
				cell.setCellValue(delta1.get(i)[j]);

				cell = row.createCell(4);
				cell.setCellValue(filtred2.get(i)[j]);
				
				cell = row.createCell(5);
				cell.setCellValue(delta2.get(i)[j]);

				cell = row.createCell(6);
				cell.setCellValue(filtred3.get(i)[j]);
				
				cell = row.createCell(7);
				cell.setCellValue(delta3.get(i)[j]);
				
				cell = row.createCell(8);
				cell.setCellValue(filtred4.get(i)[j]);
			}
		}
		
		try {
		    FileOutputStream out = 
		            new FileOutputStream(new File(fileName));
		    workbook.write(out);
		    out.close();
		     
		} catch (IOException e) {
		    e.printStackTrace();
		}
	}
	
	public void writeTestRun(String fileName, int i) throws Exception {
		DateFormat df = DateFormat.getDateInstance(DateFormat.SHORT);
		HSSFWorkbook workbook;
		HSSFSheet sheet;
		
		double[] out1 = predTest(filtred1.get(i),15,8, names.get(i) + " 1");
		double[] out2 = predTest(filtred2.get(i),20,10, names.get(i) + " 2");
		double[] out3 = predTest(filtred3.get(i),27,15, names.get(i) + " 3");
		double[] out4 = predTest(filtred4.get(i),35,18, names.get(i) + " 4");
		
		try {
			FileInputStream file = new FileInputStream(new File(fileName));
		    workbook = new HSSFWorkbook(file);
		} catch(IOException e) {
			workbook = new HSSFWorkbook();
		}
		sheet = workbook.getSheet(names.get(i));
		if(sheet == null) {
			sheet = workbook.createSheet(names.get(i));
		}
		
		Row row;
		Cell cell;
		
		row = sheet.createRow(0);
		
		cell = row.createCell(0);
		cell.setCellValue("Date");
		
		cell = row.createCell(1);
		cell.setCellValue("Data");

		cell = row.createCell(2);
		cell.setCellValue("Filtred 1");
		
		cell = row.createCell(3);
		cell.setCellValue("Pred 1");
		
		cell = row.createCell(4);
		cell.setCellValue("Delta 1");
		
		cell = row.createCell(5);
		cell.setCellValue("Filtred 2");
		
		cell = row.createCell(6);
		cell.setCellValue("Pred 2");
		
		cell = row.createCell(7);
		cell.setCellValue("Delta 2");

		cell = row.createCell(8);
		cell.setCellValue("Filtred 3");
		
		cell = row.createCell(9);
		cell.setCellValue("Pred 3");
		
		cell = row.createCell(10);
		cell.setCellValue("Delta 3");
		
		cell = row.createCell(11);
		cell.setCellValue("Filtred 4");
		
		cell = row.createCell(12);
		cell.setCellValue("Pred 4");
		
		for(int j = 0; j < dateVal.get(i).length; ++j) {
			row = sheet.createRow(j+1);
			
			cell = row.createCell(0);
			cell.setCellValue(df.format(dateVal.get(i)[j]));
			
			cell = row.createCell(1);
			cell.setCellValue(dataList.get(i)[j]);

			cell = row.createCell(2);
			cell.setCellValue(filtred1.get(i)[j]);
			
			cell = row.createCell(3);
			cell.setCellValue(out1[j]);
			
			cell = row.createCell(4);
			cell.setCellValue(delta1.get(i)[j]);

			cell = row.createCell(5);
			cell.setCellValue(filtred2.get(i)[j]);
			
			cell = row.createCell(6);
			cell.setCellValue(out2[j]);
			
			cell = row.createCell(7);
			cell.setCellValue(delta2.get(i)[j]);

			cell = row.createCell(8);
			cell.setCellValue(filtred3.get(i)[j]);
			
			cell = row.createCell(9);
			cell.setCellValue(out3[j]);
			
			cell = row.createCell(10);
			cell.setCellValue(delta3.get(i)[j]);
			
			cell = row.createCell(11);
			cell.setCellValue(filtred4.get(i)[j]);
			
			cell = row.createCell(12);
			cell.setCellValue(out4[j]);
		}
		
		try {
		    FileOutputStream outFile = 
		            new FileOutputStream(new File(fileName));
		    workbook.write(outFile);
		    outFile.close();
		     
		} catch (IOException e) {
		    e.printStackTrace();
		}
	}
	
	private double[] predTest(double[] data, int l1, int l2, String fName) throws Exception {
		NeuroNet net;
		Train tga;
		
		double[] dt = data;
		double[] in;
		double[] out = new double[dt.length];
		
		double maxValue;
		double minValue;
		
		maxValue = MyMath.getMaxInArr(dt) * 1.3;
		minValue = MyMath.getMaxInArr(dt);
		if(minValue < 0)
			minValue *= 1.3;
		else
			minValue = 0;
		
		net = new NeuroNet(minValue,maxValue,new int[] {l1,l2,1}, new int[] {1,2,3},30,"E:\\nn\\"+ fName + ".txt");
		tga = new TrainGA(net);
		
		System.out.println("train ...");
		
		if(net.isTrain) {
			tga.addTrainNet(dt);
		} else {
			tga.trainNet(dt);
		}

		
		for(int j = 0; j < net.getNumOfInputs(); ++j) {
			out[j] = dt[j];
		}
		for(int j = 0; j < (dt.length - net.getNumOfInputs() - net.getNumOfOutputs() + 1); ++j) {
			in = MyMath.getSubArr(j,net.getNumOfInputs(),dt);
			out[j + net.getNumOfInputs()] = net.runNet(in)[0];
		}
		
		return out;
	}
}

