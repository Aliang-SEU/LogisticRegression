
/**
 * @author ALiang
 * @date 2018/06/14
 */

import java.util.Scanner;

public class LogisticRegression {

    private double[][] trainData;

    private double[] label;

    //样本数量
    private int sampleNum;

    //模型参数
    private double[] parameters;

    //权重更新参数
    private double sigma = 0.001;

    public void readData(String filename) {
        Scanner sc = new Scanner(filename);

    }

    public void readData(double[][] fileData) {
        if (fileData == null) {
            throw new RuntimeException("训练数据无效!");
        }
        sampleNum = fileData.length;
        trainData = fileData;
    }


    public void train(double[][] data, int iters) {
        for (int i = 0; i < iters; i++) {

            double[] predict = forward(data);

            backward(data, predict);
        }
        double[] predict = forward(data);
        for(int i = 0; i < data.length; i++){
            System.out.println(predict[i] + " " + label[i]);
        }
    }

    /**
     * 前向更新
     *
     * @param data
     * @return
     */
    public double[] forward(double[][] data) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = sigmoid(forwardEachSample(data[i]));
        }
        return result;
    }


    /**
     * 梯度下降求取参数
     *
     * @param data
     * @param predict
     */
    public void backward(double[][] data, double[] predict) {


        for (int i = 0; i < parameters.length; i++) {
            double f = 0.0;
            for (int j = 0; j < sampleNum; j++) {
                f += (label[j] - predict[j]) * data[i][j];
            }
            parameters[i] += sigma * f;
        }

    }

    public double forwardEachSample(double[] a) {
        double sum = 0.0d;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * parameters[i];
        }
        return sum;
    }

    public double sigmoid(double a) {
        return 1 / (1 + Math.exp(-a));
    }

    public double calcError(double[] predict) {

        double error = 0.0d;

        for (int i = 0; i < predict.length; i++) {
            error += Math.abs(predict[i] - label[i]);
        }

        return error / predict.length;
    }


    public static void main(String[] args) {

    }
}
