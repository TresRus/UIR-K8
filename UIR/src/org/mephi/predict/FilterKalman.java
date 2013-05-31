package org.mephi.predict;

public class FilterKalman {
	private double X0; // predicted state
	private double P0; // predicted covariance

    private double F; // factor of real value to previous real value
    private double Q; // measurement noise
    private double H; // factor of measured value to real value
    private double R; // environment noise

    private double State;
    private double Covariance;

    public FilterKalman(double q, double r) {
        Q = q;
        R = r;
        F = 1;
        H = 1;
    }
    
    public FilterKalman(double q, double r, double f, double h) {
        Q = q;
        R = r;
        F = f;
        H = h;
    }

    public void setState(double state, double covariance) {
        State = state;
        Covariance = covariance;
    }

    public double correct(double data) {
        //time update - prediction
        X0 = F*State;
        P0 = F*Covariance*F + Q;

        //measurement update - correction
        double K = H*P0/(H*P0*H + R);
        State = X0 + K*(data - H*X0);
        Covariance = (1 - K*H)*P0;   
        return State;
    }
    
    public double[] runFilter(double[] data) {
    	double[] res = new double[data.length];
    	
    	setState((data[0] + data[1] + data[2])/3, 0.05);
    	for(int i = 0; i < data.length; ++i) {
    		res[i] = correct(data[i]);
    	}
    	
    	return res;
    }
}
