package org.mephi.predict;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;

public class DataIO {
	
	public static List<double[]> readData(String fileName) throws IOException {
		List<double[]> res = new ArrayList<double[]>();
		double[] atmdata;
		List<Double> ad = new ArrayList<Double>();
		FileInputStream file = new FileInputStream(new File(fileName));
		
		HSSFWorkbook workbook = new HSSFWorkbook(file);
		
		HSSFSheet sheet = workbook.getSheetAt(1);
		
		Iterator<Row> rowIterator = sheet.iterator();
		Row row = rowIterator.next();
		double prevATMnum = 0.0;
		double summ;
		while(rowIterator.hasNext()) {
			row = rowIterator.next();
			
			Cell ATMcell = row.getCell(0);
			Cell summCell = row.getCell(4);
			summ = summCell.getNumericCellValue();
			if(prevATMnum == ATMcell.getNumericCellValue()) {
				ad.add(summ);
			} else {
				atmdata = new double[ad.size()];
				for(int i = 0; i < ad.size(); ++i) {
					atmdata[i] = ad.get(i).doubleValue();
				}
				res.add(atmdata);
				ad = new ArrayList<Double>();
				ad.add(summ);
			}
		}
		
		atmdata = new double[ad.size()];
		for(int i = 0; i < ad.size(); ++i) {
			atmdata[i] = ad.get(i).doubleValue();
		}
		res.add(atmdata);
		
		return res;
	}
}
