package com.company;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

public class Main {

    public static int[] get_image_pixel(String image_path)
    {
        int[] rgb = new int[3];
        File file = new File(image_path);
        BufferedImage bi = null;
        try {
            bi = ImageIO.read(file);
        } catch (Exception e) {
            e.printStackTrace();
        }

        int width = bi.getWidth();
        int height = bi.getHeight();
        Raster raster = bi.getData();
        int [] temp = new int[raster.getWidth()*raster.getHeight()*raster.getNumBands()];
        int [] data  = raster.getPixels(0,0,raster.getWidth(),raster.getHeight(),temp);

        return data;

    }
    public static ArrayList<String> read_valida_index_file() {
        System.out.println("read valida index ");
        File file = new File("/home/ices/PycharmProject/MultistageConvRNN/data/evaluate/valid_test.txt");
        BufferedReader reader = null;
        ArrayList<String> list = new ArrayList<String>();
        try {
            reader = new BufferedReader(new FileReader(file));
            String tempStr;
            while ((tempStr = reader.readLine()) != null) {
                int index = Integer.parseInt(tempStr);
//                System.out.println("index is:"+String.valueOf(index));
                list.add(String.valueOf(index));
            }
//            for (int i = 0; i < list.size(); i++) {
//                System.out.println("Index: " + i + " - Item: " + list.get(i));
//            }

            reader.close();
            return list;
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        }

        return list;


    }
    public static void print_list(double [] data,Score score,String eva_index){
        for(int i=0;i<data.length;i++){
            System.out.println(data[i]);
        }
        System.out.println("the average of "+eva_index+" is:"+score.get_average(data));
    }

    public double[][] avg(double[][] data,int[] nan_num){

        for (int i = 0;i<data.length;i++){
            for (int j = 0;j<data[i].length;j++){
                data[i][j] = data[i][j]/(4000-nan_num[i]);
            }
        }
        return data;

    }
    public double[][][] evaluate_sequence(String evaluate_fold){
        double[][][] result = new double[4][10][3];
        System.out.println("Evaluate "+evaluate_fold);
        Score score = new Score();
        String root_real_path = "/mnt/A/CIKM2017/CIKM_datasets/test/";
        String root_pre_path = "/mnt/A/meteorological/2500_ref_seq/"+evaluate_fold+"/";

        double[][] current_sample_hss = new double[10][3];
        double[][] current_sample_csi = new double[10][3];
        double[][] current_sample_pod = new double[10][3];
        double[][] current_sample_far = new double[10][3];
        int[] nan_num = new int[10];
        int discard_sample = 0;
        for (int i = 1; i <4001; i++){

            String real_fold = root_real_path+"sample_"+String.valueOf(i)+"/";
            String pre_fold = root_pre_path+"sample_"+String.valueOf(i)+"/";


            for (int j=6; j<16;j++){
                String real_img_path = real_fold + "img_"+String.valueOf(j)+".png";
                String pre_img_path = pre_fold + "img_"+String.valueOf(j)+".png";
                int[] pred_img = get_image_pixel(pre_img_path);
                int[] real_img = get_image_pixel(real_img_path);
                score.calculate(pred_img,real_img);
                if (score.judeg_NaN(score.getHss())||score.judeg_NaN(score.getCsi())||score.judeg_NaN(score.getPod())||score.judeg_NaN(score.getFar())){
                    nan_num[j-6]++;
                    continue;
                }
                current_sample_hss[j-6] = score.getHss();
                current_sample_csi[j-6] = score.getCsi();
                current_sample_pod[j-6] = score.getPod();
                current_sample_far[j-6] = score.getFar();
            }
        }
        current_sample_hss = avg(current_sample_hss,nan_num);
        current_sample_csi = avg(current_sample_csi,nan_num);
        current_sample_pod = avg(current_sample_pod,nan_num);
        current_sample_far = avg(current_sample_far,nan_num);

        result[0] = current_sample_hss;
        result[1] = current_sample_csi;
        result[2] = current_sample_pod;
        result[3] = current_sample_far;


        return result;

    }

    public double[][] evaluate(String evaluate_fold){
        double[][] result = new double[4][3];
        System.out.println("Evaluate "+evaluate_fold);
        Score score = new Score();
        String root_real_path = "/mnt/A/CIKM2017/CIKM_datasets/test/";
        String root_pre_path = "/mnt/A/meteorological/2500_ref_seq/"+evaluate_fold+"/";

        double[] hss = new double[3];
        double[] csi = new double[3];
        double[] pod = new double[3];
        double[] far = new double[3];

        int discard_sample = 0;
        for (int i = 1; i <4001; i++){

            String real_fold = root_real_path+"sample_"+String.valueOf(i)+"/";
            String pre_fold = root_pre_path+"sample_"+String.valueOf(i)+"/";
            double[] current_sample_hss = new double[3];
            double[] current_sample_csi = new double[3];
            double[] current_sample_pod = new double[3];
            double[] current_sample_far = new double[3];
            int nan_num = 0;
            for (int j=6; j<16;j++){
                String real_img_path = real_fold + "img_"+String.valueOf(j)+".png";
                String pre_img_path = pre_fold + "img_"+String.valueOf(j)+".png";
                int[] pred_img = get_image_pixel(pre_img_path);
                int[] real_img = get_image_pixel(real_img_path);
                score.calculate(pred_img,real_img);
                if (score.judeg_NaN(score.getHss())||score.judeg_NaN(score.getCsi())||score.judeg_NaN(score.getPod())||score.judeg_NaN(score.getFar())){
                    nan_num++;
                    continue;
                }
                current_sample_hss = score.additive(current_sample_hss,score.getHss());
                current_sample_csi = score.additive(current_sample_csi,score.getCsi());
                current_sample_pod= score.additive(current_sample_pod,score.getPod());
                current_sample_far = score.additive(current_sample_far,score.getFar());
            }
            current_sample_hss = score.average(current_sample_hss,10-nan_num);
            current_sample_csi = score.average(current_sample_csi,10-nan_num);
            current_sample_pod = score.average(current_sample_pod,10-nan_num);
            current_sample_far = score.average(current_sample_far,10-nan_num);

            if (nan_num==10){
                discard_sample++;
                continue;

            }


            hss = score.additive(hss,current_sample_hss);
            csi = score.additive(csi,current_sample_csi);
            pod= score.additive(pod,current_sample_pod);
            far = score.additive(far,current_sample_far);
        }
        hss = score.average(hss,4000-discard_sample);
        csi = score.average(csi,4000-discard_sample);
        pod = score.average(pod,4000-discard_sample);
        far = score.average(far,4000-discard_sample);
        for(int i=0;i<hss.length;i++)
        {
            result[0][i] = hss[i];
            result[1][i] = csi[i];
            result[2][i] = pod[i];
            result[3][i] = far[i];
        }

        return result;

    }

    public double[][] evaluate_seq(String evaluate_fold,String type){
        // write your code here

        System.out.println("Evaluate "+evaluate_fold+"  type is:"+type);
        Score score = new Score();
        String root_real_path = "/mnt/A/CIKM2017/CIKM_datasets/test/";
        String root_pre_path = "/mnt/A/meteorological/2500_ref_seq/"+evaluate_fold+"/";


        double[][] res = new double[10][3];

        int[] nan_nums = new int[10];
        int discard_sample = 0;

        for (int i = 1; i <4001; i++){
//            System.out.println("evaluation complete  "+String.valueOf(100.0*i/4000.0)+" %");
            String real_fold = root_real_path+"sample_"+String.valueOf(i)+"/";
            String pre_fold = root_pre_path+"sample_"+String.valueOf(i)+"/";
            int nan_num = 0;
            for (int j=6; j<16;j++){
                String real_img_path = real_fold + "img_"+String.valueOf(j)+".png";
                String pre_img_path = pre_fold + "img_"+String.valueOf(j)+".png";
                int[] pred_img = get_image_pixel(pre_img_path);
                int[] real_img = get_image_pixel(real_img_path);
                score.calculate(pred_img,real_img);
                if (type.equals("csi")) {
                    if (score.judeg_NaN(score.getCsi())) {
                        nan_nums[j - 6]++;
                        continue;
                    }
                    res[j - 6] = score.additive(res[j - 6], score.getCsi());
                }else{
                    if (type.equals("hss")){
                        if (score.judeg_NaN(score.getHss())) {
                            nan_nums[j - 6]++;
                            continue;
                        }
                        res[j - 6] = score.additive(res[j - 6], score.getHss());
                    }
                    else{
                        if(type.equals("pod")){
                            if (score.judeg_NaN(score.getPod())) {
                                nan_nums[j - 6]++;
                                continue;
                            }
                            res[j - 6] = score.additive(res[j - 6], score.getPod());
                        }
                        else {
                            if (score.judeg_NaN(score.getFar())) {
                                nan_nums[j - 6]++;
                                continue;
                            }
                            res[j - 6] = score.additive(res[j - 6], score.getFar());
                        }
                    }
                }

            }

        }
        for (int i = 0;i<10;i++){
            res[i] = score.average(res[i],4000-nan_nums[i]);
//            print_list(res[i],score,"CSI at "+String.valueOf(i)+" time");
        }
        return res;

    }

    public static void main(String[] args) {
        //// write your code here

        String evaluate_fold = args[0];
//    String evaluate_fold = "CIKM_dec_ConvGRU_test";
        System.out.println("Evaluate "+evaluate_fold);
    Score score = new Score();
    String root_real_path = "/mnt/A/CIKM2017/CIKM_datasets/test/";
    String root_pre_path = "/mnt/A/meteorological/2500_ref_seq/"+evaluate_fold+"/";
    ArrayList<String> valid_indexes = read_valida_index_file();
    double[] hss = new double[3];
    double[] csi = new double[3];
    double[] pod = new double[3];
    double[] far = new double[3];


//    for (int ind = 0; ind < valid_indexes.size(); ind++) {
//        int i = Integer.parseInt(valid_indexes.get(ind));
//        String real_fold = root_real_path+"sample_"+String.valueOf(i)+"/";
//        String pre_fold = root_pre_path+"sample_"+String.valueOf(i)+"/";
//        for (int j=6; j<16;j++){
//            String real_img_path = real_fold + "img_"+String.valueOf(j)+".png";
//            String pre_img_path = pre_fold + "img_"+String.valueOf(j)+".png";
//            int[] pred_img = get_image_pixel(pre_img_path);
//            int[] real_img = get_image_pixel(real_img_path);
//            score.calculate(pred_img,real_img);
//        }
//    }
    for (int i = 1; i <4001; i++){
        String real_fold = root_real_path+"sample_"+String.valueOf(i)+"/";
        String pre_fold = root_pre_path+"sample_"+String.valueOf(i)+"/";
        for (int j=6; j<16;j++){
            String real_img_path = real_fold + "img_"+String.valueOf(j)+".png";
            String pre_img_path = pre_fold + "img_"+String.valueOf(j)+".png";
            int[] pred_img = get_image_pixel(pre_img_path);
            int[] real_img = get_image_pixel(real_img_path);
            score.calculate(pred_img,real_img);
            }
        }
    score.get_result();
    print_list(score.getHss(),score,"HSS");
    print_list(score.getCsi(),score,"CSI");
    print_list(score.getPod(),score,"POD");
    print_list(score.getFar(),score,"FAR");

    }

}
