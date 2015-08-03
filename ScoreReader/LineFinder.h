
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PI 3.1415926

class LineFinder {
private:
    cv::Mat img; // 원 영상
    std::vector<cv::Vec4i> lines; // 선을 감지하기 위한 마지막 점을 포함한 벡터
    double deltaRho;
    double deltaTheta; // 누산기 해상도 파라미터
    int minVote; // 선을 고려하기 전에 받아야 하는 최소 투표 개수
    double minLength; // 선에 대한 최소 길이
    double maxGap; // 선에 따른 최대 허용 간격

public:
    LineFinder();
    ~LineFinder();
    // 기본 누적 해상도는 1각도 1화소 
    // 간격이 없고 최소 길이도 없음

    // 누적기에 해상도 설정
    void setAccResolution(double dRho, double dTheta);

    // 투표 최소 개수 설정
    void setMinVote(int minv);

    // 선 길이와 간격 설정
    void setLineLengthAndGap(double length, double gap);

    // 허프 선 세그먼트 감지를 수행하는 메소드
    // 확률적 허프 변환 적용
    std::vector<cv::Vec4i> findLines(cv::Mat& binary);

    // 위 메소드에서 감지한 선을 다음 메소드를 사용해서 그림
    // 영상에서 감지된 선을 그리기
    void drawDetectedLines(cv::Mat &image, cv::Scalar color = cv::Scalar(255, 255, 255));
};