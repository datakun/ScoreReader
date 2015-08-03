#include "LineFinder.h"


LineFinder::LineFinder()
    : deltaRho(1), deltaTheta(PI / 180), minVote(10), minLength(0.), maxGap(0.)
{
}


LineFinder::~LineFinder()
{
}

void LineFinder::setAccResolution(double dRho, double dTheta) {
    deltaRho = dRho;
    deltaTheta = dTheta;
}

void LineFinder::setMinVote(int minv) {
    minVote = minv;
}

void LineFinder::setLineLengthAndGap(double length, double gap) {
    minLength = length;
    maxGap = gap;
}

std::vector<cv::Vec4i> LineFinder::findLines(cv::Mat& binary) {
    lines.clear();
    cv::HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);
    return lines;
} // cv::Vec4i 벡터를 반환하고, 감지된 각 세그먼트의 시작과 마지막 점 좌표를 포함.

void LineFinder::drawDetectedLines(cv::Mat &image, cv::Scalar color) {

    // 선 그리기
    std::vector<cv::Vec4i>::const_iterator it2 = lines.begin();

    while (it2 != lines.end()) {
        cv::Point pt1((*it2)[0], (*it2)[1]);
        cv::Point pt2((*it2)[2], (*it2)[3]);
        cv::line(image, pt1, pt2, color);
        ++it2;
    }
}