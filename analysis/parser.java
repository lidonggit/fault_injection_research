import java.text.DecimalFormat;
import java.util.*;
import java.io.*;

public class parser {
	
	private static DecimalFormat df = new DecimalFormat( "0.0000" );
	private static String fileName_log = "";
	private static String fileName_trace = "";
	private static String fileName_out = "";
	private static Vector<Double> timeRand = new Vector<Double>();
	private static Vector<Double> spacRand = new Vector<Double>();
	private static double orig_exec_time;
	private static Vector<Double> stTime = new Vector<Double>();
	private static Vector<Double> btTime = new Vector<Double>();
	private static Vector<Double> exTime = new Vector<Double>();
	private static Vector<String> verify = new Vector<String>();
	private static Vector<String> lineNum = new Vector<String>();
	private static double perf_fault_threshold = 0.1;
	
    public static void main (String[] args) {
    	
    	if(args.length!=2) {
    		System.out.println("Need the input file names: java parser [logfile] [tracefile]");
    		System.exit(0);
    	}
    	else {
    		fileName_log = args[0];
    		fileName_trace = args[1];
    		fileName_out = args[0].substring(0,5)+"out.txt";
    		System.out.println("Input File Names: "+ fileName_log+" "+fileName_trace);
    		System.out.println("Output File Name: "+fileName_out);
    	}

    	String oneLine = "";
    	String oneItem = "";
    	
    	try {
    		/* Extract information from the log file */
			Scanner sc = new Scanner(new File(fileName_log));
			while(sc.hasNextLine()) {
				oneLine = sc.nextLine();
				/* Get the sapce randomness */
				if(oneLine.startsWith("randomness")) {
					Scanner lineSc = new Scanner(oneLine).useDelimiter("\\s+");
					while(lineSc.hasNext()) {
						oneItem = lineSc.next();
						if(oneItem.equals("randomness:")) {
							timeRand.addElement(lineSc.nextDouble());
							spacRand.addElement(lineSc.nextDouble());
						}
					}
					lineSc.close();
				}
				/* Get the execution time without faults */
				if(oneLine.startsWith("[FI] exec_time=")) {
					Scanner lineSc = new Scanner(oneLine).useDelimiter("=|,");
					while(lineSc.hasNext()) {
						oneItem = lineSc.next();
						oneItem = lineSc.next();
						orig_exec_time = Double.parseDouble(oneItem);
						break;
					}
					lineSc.close();
				}
				/* Get the launch time of fault injection */
				if(oneLine.startsWith("Launch Time")) {
					Scanner lineSc = new Scanner(oneLine).useDelimiter("\\s+");
					while(lineSc.hasNext()) {
						oneItem = lineSc.next();
						if(oneItem.equals("Time:")) {
							stTime.addElement(lineSc.nextDouble());
						}
					}
					lineSc.close();
				}
				/* Get the process time of fault injection */
				if(oneLine.startsWith("Process Time")) {
					Scanner lineSc = new Scanner(oneLine).useDelimiter("\\s+");
					while(lineSc.hasNext()) {
						oneItem = lineSc.next();
						if(oneItem.equals("Time:")) {
							btTime.addElement(lineSc.nextDouble());
						}
					}
					lineSc.close();
				}
				/* Get the execution time of whole application */
				if(oneLine.startsWith("Execution Time")||oneLine.startsWith("Time in seconds")) {
					Scanner lineSc = new Scanner(oneLine).useDelimiter("\\s+");
					while(lineSc.hasNext()) {
						oneItem = lineSc.next();
						if(oneItem.equals("Time:")||oneItem.equals("=")) {
							exTime.addElement(lineSc.nextDouble());
						}
					}
					lineSc.close();
				}
				/* Get the verification result of the application */
				if(oneLine.startsWith("VERIFICATION")||oneLine.startsWith("Verification")) {
					Scanner lineSc = new Scanner(oneLine).useDelimiter("\\s+");
					while(lineSc.hasNext()) {
						oneItem = lineSc.next();
						if(oneItem.equals("VERIFICATION")||oneItem.equals("FT")) {
							verify.addElement(lineSc.next());
						}
					}
					lineSc.close();
				}
			}
			sc.close();
			
			/* Extract information from the trace file */
			Scanner sc1 = new Scanner(new File(fileName_trace));
			while(sc1.hasNextLine()) {
				oneLine = sc1.nextLine();
				/* Get the line number before injection */
				if(oneLine.contains("signal handler called")) {
					String lastLine = sc1.nextLine();
					Scanner lineSc = new Scanner(lastLine).useDelimiter("\\s+");
					while(lineSc.hasNext()) {
						oneItem = lineSc.next();
					}
					lineNum.addElement(oneItem);
					lineSc.close();
				}
			}
			sc1.close();
			
			/* Write the results to a file */
	
			BufferedWriter out=new BufferedWriter(new FileWriter(fileName_out));
			out.write("Time_Random Time_Random_Adjust Space_Random Result Performance_Result Line_Number" + "\n");
			for(int i=0; i<spacRand.size(); i++) {
				String performance_fault = "PSUCC";
				if(Math.abs(exTime.elementAt(i)-orig_exec_time)/orig_exec_time>perf_fault_threshold) {
					performance_fault = "PFAIL";
				}
	        	double timeRand_adjust = stTime.elementAt(i) / (exTime.elementAt(i)*1000000-btTime.elementAt(i));
//	        	System.out.println(df.format(timeRand.elementAt(i))+" "+df.format(timeRand_adjust)+" "+df.format(spacRand.elementAt(i))+" "+
//	        			  verify.elementAt(i).substring(0,6) + " " + lineNum.elementAt(i) + "\n");
	        	out.write(df.format(timeRand.elementAt(i))+" "+df.format(timeRand_adjust)+" "+df.format(spacRand.elementAt(i))+" "+
	        			  verify.elementAt(i).substring(0,6) + " " + performance_fault + " " + lineNum.elementAt(i) +"\n");
			}
			out.close();
			
    	} catch(Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
    }
}
