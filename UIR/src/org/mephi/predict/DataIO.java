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
		
		FileInputStream file = new FileInputStream(new File(fileName));
		
		HSSFWorkbook workbook = new HSSFWorkbook(file);
		
		HSSFSheet sheet = workbook.getSheetAt(1);
		
		Iterator<Row> rowIterator = sheet.iterator();
		Row row = rowIterator.next();
		double prevATMnum;
		//while(rowIterator.hasNext()) {
			row = rowIterator.next();
			
			Cell ATMcell = row.getCell(0);
			Cell summCell = row.getCell(4);
			double summ;
			switch(ATMcell.getCellType()) {
            case Cell.CELL_TYPE_BOOLEAN:
                System.out.print("bool");
                break;
            case Cell.CELL_TYPE_NUMERIC:
                System.out.print("num");
                break;
            case Cell.CELL_TYPE_STRING:
            	System.out.print("summ");
                break;
			}
		//}
		
		return res;
	}
}
