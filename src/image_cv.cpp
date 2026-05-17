/*
 * @Author: ilikara 3435193369@qq.com
 * @Date: 2025-01-04 06:50:56
 * @LastEditors: ilikara 3435193369@qq.com
 * @LastEditTime: 2025-03-13 08:15:10
 * @FilePath: /2k300_smartcar/src/image_cv.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "image_cv.h"
#include <stdio.h>
#include <fstream>
#include "global.h"

#define calc_scale 2

cv::Mat raw_frame;
cv::Mat grayFrame;
cv::Mat binarizedFrame;
cv::Mat morphologyExFrame;
cv::Mat track;
cv::Mat resized_raw_Frame;

std::vector<int> left_line;              // 左边缘列号数组
std::vector<int> right_line;             // 右边缘列号数组
std::vector<int> mid_line;               // 中线列号数组

std::vector<int> left_line_filtered;     // 滤波后的左边缘列号数组
std::vector<int> right_line_filtered;    // 滤波后的右边缘列号数组
std::vector<int> mid_line_filtered;      // 滤波后的中线列号数组

int line_tracking_width;
int line_tracking_height;
double lthzs=0.7;
int mid_pro=0;
cv::Mat image_binerize(cv::Mat &frame)
{
    cv::Mat output;
    cv::Mat binarizedFrame;
    cv::Mat hsvImage;
    cv::cvtColor(frame, hsvImage, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);
    cv::threshold(hsvChannels[0], binarizedFrame, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    cv::threshold(hsvChannels[1], output, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    cv::bitwise_or(output, binarizedFrame, output);
    return output;
}

cv::Mat find_road(cv::Mat &frame)
{
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2, 2));
    cv::morphologyEx(binarizedFrame, morphologyExFrame, cv::MORPH_OPEN, kernel);
    cv::Mat mask = cv::Mat::zeros(line_tracking_height + 2, line_tracking_width + 2, CV_8UC1);
    cv::Point seedPoint(line_tracking_width / 2, line_tracking_height - 10);
    cv::circle(morphologyExFrame, seedPoint, 10, 255, -1);
    cv::Scalar newVal(128);
    cv::Scalar loDiff = cv::Scalar(15);
    cv::Scalar upDiff = cv::Scalar(25);

    cv::floodFill(morphologyExFrame, mask, seedPoint, newVal, 0, loDiff, upDiff, 8);

    cv::Mat outputImage = cv::Mat::zeros(line_tracking_width, line_tracking_height, CV_8UC1);

    mask(cv::Rect(1, 1, line_tracking_width, line_tracking_height)).copyTo(outputImage);

    return outputImage;
}

void image_main()
{
    cv::resize(raw_frame, resized_raw_Frame, cv::Size(line_tracking_width, line_tracking_height));
    binarizedFrame = image_binerize(resized_raw_Frame);
    track = find_road(binarizedFrame);
    left_line.clear();
    right_line.clear();
    mid_line.clear();
  
    left_line.resize(line_tracking_height, -1);
    right_line.resize(line_tracking_height, -1);
    mid_line.resize(line_tracking_height, -1);
    
    left_line_filtered.resize(line_tracking_height, -1);
    right_line_filtered.resize(line_tracking_height, -1);
    mid_line_filtered.resize(line_tracking_height, -1);
  
    uchar(*IMG)[line_tracking_width] = reinterpret_cast<uchar(*)[line_tracking_width]>(track.data);

    // 预计算常量
    const double LEFT_WEIGHT_3_5 = 3.0/5.0;
    const double RIGHT_WEIGHT_2_5 = 2.0/5.0;
    const double LEFT_WEIGHT_2_5 = 2.0/5.0;
    const double RIGHT_WEIGHT_3_5 = 3.0/5.0;

    // 第一阶段：找边界
    #pragma omp parallel for
    for (int i = 0; i < line_tracking_height; i += 1) {  // 降采样
        int max_start = -1, max_end = -1;
        int current_start = -1, current_length = 0;
        int max_length = 0;
        
        for (int j = 0; j < line_tracking_width; ++j) {
            if (IMG[i][j]) {
                if (!current_length) current_start = j;
                current_length++;
                if (current_length >= max_length) {
                    max_length = current_length;
                    max_start = current_start;
                    max_end = j;
                }
            } else {
                if (max_length > line_tracking_width / 2) break;
                current_length = 0;
            }
        }
        
        left_line[i] = max_length > 0 ? max_start : -1;
        right_line[i] = max_length > 0 ? max_end : -1;
    }

    // 第二阶段：计算中线
    for (int row = line_tracking_height - 1; row >= 10; --row) {
        int left = left_line[row];
        int right = right_line[row];
        
        if (left == -1 && right == -1) {
            mid_line[row] = mid_line[row + 1];
        
        } else {
            // if(left < 10 && right <= 65)
            //     mid_line[row] = left*LEFT_WEIGHT_3_5 + right*RIGHT_WEIGHT_2_5;
            if(right > 75&&5<left<40 )
                mid_line[row] = left*LEFT_WEIGHT_2_5 + right*RIGHT_WEIGHT_3_5;
            else if (left<10&&75>right>40)
                mid_line[row] = left*RIGHT_WEIGHT_3_5 + right*RIGHT_WEIGHT_2_5;
             else   
                mid_line[row] = (left + right) * 0.5+2;
        }
    }
    // 第三阶段：滤波处理
    const double a = 0.3; // 滤波系数，可以根据需要调整
    // 初始化滤波数组的边界值
    left_line_filtered[line_tracking_height-1] = static_cast<double>(left_line[line_tracking_height-1]);
    right_line_filtered[line_tracking_height-1] = static_cast<double>(right_line[line_tracking_height-1]);
    mid_line_filtered[line_tracking_height-1] = static_cast<double>(mid_line[line_tracking_height-1]);
    
    left_line_filtered[0] = static_cast<double>(left_line[0]);
    right_line_filtered[0] = static_cast<double>(right_line[0]);
    mid_line_filtered[0] = static_cast<double>(mid_line[0]);
    
    // 应用滤波
    for (int row = line_tracking_height - 2; row > 0; --row) {
        if (left_line[row] != -1) {
            left_line_filtered[row] = a * left_line[row] + a * left_line_filtered[row + 1] + (1 - 2 * a) * left_line_filtered[row - 1];
        } else {
            left_line_filtered[row] = left_line_filtered[row + 1];
        }        
        if (right_line[row] != -1) {
            right_line_filtered[row] = a * right_line[row] + a * right_line_filtered[row + 1] + (1 - 2 * a) * right_line_filtered[row - 1];
        } else {
            right_line_filtered[row] = right_line_filtered[row + 1];
        }
        if (mid_line[row] != -1) {
            mid_line_filtered[row] = a * mid_line[row] + (1 - a) * mid_line_filtered[row + 1];
        } else {
            mid_line_filtered[row] = mid_line_filtered[row + 1];
        }
        //mid_line_filtered[row]+=2;
    }

    // 使用滤波后的值进行输出
    // 打印图像尺寸信息
    printf("图像处理尺寸: %d x %d\n", line_tracking_width, line_tracking_height);
    // 打印前瞻点那一行的车道线坐标
    double foresee_y = readDoubleFromFile(foresee_file);
    int foresee_row = static_cast<int>(foresee_y / calc_scale);  // 转换为行索引

    // mid_line[foresee_row]=lthzs*mid_line[foresee_row]+(1-lthzs)*mid_pro;
    // mid_pro=mid_line[foresee_row];

    // 确保行索引在有效范围内
    if (foresee_row >= 0 && foresee_row < line_tracking_height) {
        printf("前瞻点行%d: 左线=%d, 右线=%d, 中线=%d,\n",
               foresee_row,
               left_line_filtered[foresee_row],
               right_line_filtered[foresee_row],
               mid_line_filtered[foresee_row]);
        }
}