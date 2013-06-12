package org.mephi.predict;

public class Predict {
	
	public static void main(String[] args) {
		try {
			DataIO data = new DataIO();
			
			data.readData("E:\\data.xls");
			
			for(int i = 0; i < data.dataList.size(); ++i) {
				System.out.printf("Run for data #%d :\n", i);
				
				if(i == 14) {
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
