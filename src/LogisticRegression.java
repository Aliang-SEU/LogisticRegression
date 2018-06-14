
/**
 * @author ALiang
 * @date 2018/06/14
 */

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class LogisticRegression {

    //训练数据
    private double[][] trainData;

    //样本的标签 二分类为 0 或 1
    private double[] label;

    //每次预测的值
    private double[] predict;

    //样本数量
    private int sampleNum;

    //样本的特征维度
    private int sampleDim;

    //模型参数 这里假设是线性模型 y = wx + b ; (b = 0)
    private double[] parameters;

    //梯度下降的步长
    private double sigma = 0.001;

    //模型停止的阈值
    private double epsilon = 1e-9;


    private void readData(double[][] trainData, double[] label) {
        if (trainData == null || label == null) {
            throw new RuntimeException("训练数据无效!");
        }

        sampleNum = trainData.length;
        sampleDim = trainData[0].length;

        //参数初始化 这里完全随机 更好可以使用一些分布
        parameters = new double[sampleDim];
        Random rand = new Random();
        for(int i = 0; i < parameters.length; i++){
            parameters[i] = rand.nextDouble();
        }

        this.trainData = trainData;
        this.label = label;
        this.predict = new double[sampleNum];
    }

    /**
     * 训练器
     * @param trainData
     * @param label
     * @param maxIters
     * @param debug
     */
    public void train(double[][] trainData, double[] label, int maxIters, boolean debug) {

        //准备数据
        readData(trainData, label);
        System.out.println("开始训练...");
        //训练
        for (int i = 0; i < maxIters ; i++) {

            //一次前向传播
            forward();

            double error =  calcError();
            if(debug){
                System.out.println("第" + i + "次的平均误差:" + error);
            }

            if(error < epsilon){
                break;
            }

            //一次反向参数更新
            backward();
        }

        System.out.println("训练完成...");
    }

    /**
     * 模型预测
     * @param data
     * @return
     */
    public double predict(double[] data){
        return sigmoid(forwardEachSample(data));
    }

    public double[] predict(double[][] data){
        double[] predict = new double[data.length];
        for(int i = 0; i < data.length; i++){
            predict[i] = sigmoid(forwardEachSample(data[i]));
        }
        return predict;
    }

    /**
     * 前向传播
     *
     * @return
     */
    private double[] forward() {

        for (int i = 0; i < sampleNum; i++) {
            predict[i] = sigmoid(forwardEachSample(trainData[i]));
        }
        return predict;
    }


    /**
     *  最大似然估计
     *  批量梯度下降求取参数
     */
    public void backward() {

        for (int i = 0; i < parameters.length; i++) {

            double f = 0.0;
            for (int j = 0; j < sampleNum; j++) {
                /**
                 * 推导公式 wj = wj + n * sigma(yi - zi) * xji;
                 */
                f += (label[j] - predict[j]) * trainData[j][i];
            }
            parameters[i] += sigma * f;
        }
    }

    /**
     * 向量相乘
     * @param a
     * @return
     */
    public double forwardEachSample(double[] a) {
        double sum = 0.0d;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * parameters[i];
        }
        return sum;
    }

    /**
     * 激活函数
     * @param a
     * @return
     */
    public double sigmoid(double a) {
        return 1 / (1 + Math.exp(-a));
    }

    /**
     * 计算每次迭代误差
     * @return
     */
    public double calcError() {

        double error = 0.0d;

        for (int i = 0; i < predict.length; i++) {
            error += Math.abs(predict[i] - label[i]);
        }

        return error / predict.length;
    }

    public double calcPredictError(double[] label, double[] predict){
        assert (label.length == predict.length);
        double sumError = 0.0d;
        int sumErrorCount = 0;
        for(int i = 0; i < label.length; i++){
            sumError += Math.abs(label[i] - predict[i]);
            if((int)label[i] != (int)Math.round(predict[i])){
                sumErrorCount ++;
            }
        }
        System.out.println("总的误差为:" + sumError);
        System.out.println("平均误测误差为:" + sumError / label.length);
        System.out.println("预测正确数目:" + (label.length - sumErrorCount) + "/" + label.length);
        System.out.println("预测正确率:" + (1 - sumErrorCount * 1.0 / label.length));
        return sumError / label.length;
    }

    /**
     * 测试程序
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(new File("src/irisData/data.txt"));
        String[] names = {"Iris-setosa","Iris-versicolor", "Iris-virginica"};
        List<String> list = new ArrayList<>();
        while(sc.hasNextLine()){
            list.add(sc.nextLine());
        }
        double[][] train_data = new double[list.size()][4];
        double[] label = new double[list.size()];
        int idx = 0;
        for(String s : list){
            String[] data = s.split(",");
            for(int i = 0; i < 4; i++){
                train_data[idx][i] = Double.parseDouble(data[i]);
            }
            if(data[4].equals(names[2])){
                label[idx] = 1;
            }else
                label[idx] = 0;
            idx++;
        }

        LogisticRegression lr = new LogisticRegression();
        lr.train(train_data, label, 50000, true);
        double[] predict = lr.predict(train_data);

        lr.calcPredictError(label, predict);


    }
}
