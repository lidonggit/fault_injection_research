import java.text.DecimalFormat;
import java.util.*;
import java.io.*;

public class parser {
	
	private static DecimalFormat df = new DecimalFormat( "0.0000" );
	private static String fileName_log = "";
	private static String fileName_trace = "";
	private static String fileName_out = "";
	private static Vector<Double> spacRand = new Vector<Double>();
	private static Vector<Double> stTime = new Vector<Double>();
	private static Vector<Double> btTime = new Vector<Double>();
	private static Vector<Double> exTime = new Vector<Double>();
	private static Vector<String> verify = new Vector<String>();
	private static Vector<String> lineNum = new Vector<String>();
	
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
							lineSc.next();
							spacRand.addElement(lineSc.nextDouble());
						}
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
				if(oneLine.startsWith("Execution Time")) {
					Scanner lineSc = new Scanner(oneLine).useDelimiter("\\s+");
					while(lineSc.hasNext()) {
						oneItem = lineSc.next();
						if(oneItem.equals("Time:")) {
							exTime.addElement(lineSc.nextDouble());
						}
					}
					lineSc.close();
				}
				/* Get the verification result of the application */
				if(oneLine.startsWith("VERIFICATION")) {
					Scanner lineSc = new Scanner(oneLine).useDelimiter("\\s+");
					while(lineSc.hasNext()) {
						oneItem = lineSc.next();
						if(oneItem.equals("VERIFICATION")) {
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
			for(int i=0; i<spacRand.size(); i++) {
	        	double timeRand = stTime.elementAt(i) / (exTime.elementAt(i)*1000000-btTime.elementAt(i));
	        	out.write(df.format(timeRand)+" "+df.format(spacRand.elementAt(i))+" "+
	        			  verify.elementAt(i).substring(0,6) + " " + lineNum.elementAt(i) + "\n");
			}
			out.close();
			
    	} catch(Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
    }
}
