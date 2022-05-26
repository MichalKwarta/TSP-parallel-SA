#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif
#pragma hd_warning_disable
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
using namespace cv;
CUDA_CALLABLE_MEMBER class Histogram
{
public:
    Mat src;
    String filepath;
    Histogram(String path)
    {
        filepath = path;
        src = imread(path);
    }
    ~Histogram()
    {
        cout << "histogram wyczyszczony" << endl;
    }
    CUDA_CALLABLE_MEMBER void createHist()
    {
        clock_t startTime = clock();
        int ostatnislesz = filepath.find_last_of("\\");
        String filename = filepath.substr(ostatnislesz + 1, filepath.length() - 1);
        printf("Tworzenie histogramu dla: %s", (char *)&filename);
        vector<Mat> bgr_planes;
        split(src, bgr_planes);
        int histSize = 256;
        float range[] = {0, 256};
        const float *histRange = {range};
        bool uniform = true;
        bool accumulate = false;
        Mat b_hist, g_hist, r_hist;
        calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange,
                 uniform, accumulate);
        calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange,
                 uniform, accumulate);
        calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange,
                 uniform, accumulate);
        int hist_w = 512;
        int hist_h = 400;
        int bin_w = cvRound((double)hist_w / histSize);
        Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        for (int i = 1; i < histSize; i++)
        {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                 Scalar(255, 0, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                 Scalar(0, 255, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                 Scalar(0, 0, 255), 2, 8, 0);
        }   
        namedWindow(filename, CV_WINDOW_AUTOSIZE);
        imshow(filename, histImage);
        clock_t stopTime = clock();
        double czas = (stopTime - startTime) / (double)CLOCKS_PER_SEC;
        // cout << "czas generacji histogramu: " << czas << "s" << endl;
        printf("czas generacji histogramu: %fs", &czas);
    }
};
__global__ void doit(Histogram *h)
{
    h->createHist();
}
int main(int argc, char **argv)
{
    vector<String> filenames;
    String folder = "C:\\Users\\chime\\Documents\\Visual Studio
        2015\\Projects\\histcpp\\x64\\Debug\\obrazy ";
        glob(folder, filenames);
    clock_t startTime = clock();
    for (size_t i = 0; i < filenames.size(); ++i)
    {
        Histogram *h = new Histogram(filenames[i]);
        Histogram *d_h;
        cudaMalloc(&d_h, sizeof(Histogram));
        cudaMemcpy(d_h, h, sizeof(Histogram), cudaMemcpyHostToDevice);
        doit<<<1, 3>>>(d_h);
        cudaMemcpy(h, d_h, sizeof(Histogram), cudaMemcpyDeviceToHost);
        cudaFree(d_h);
        delete (h);
    }
    clock_t stopTime = clock();
    double czas = (stopTime - startTime) / (double)CLOCKS_PER_SEC;
    cout << "czas generacji wszystkich histogramow: " << czas << "s" << endl;
    waitKey(0);
    getchar();
    return 0;
}