
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PI 3.1415926

class LineFinder {
private:
    cv::Mat img; // �� ����
    std::vector<cv::Vec4i> lines; // ���� �����ϱ� ���� ������ ���� ������ ����
    double deltaRho;
    double deltaTheta; // ����� �ػ� �Ķ����
    int minVote; // ���� ����ϱ� ���� �޾ƾ� �ϴ� �ּ� ��ǥ ����
    double minLength; // ���� ���� �ּ� ����
    double maxGap; // ���� ���� �ִ� ��� ����

public:
    LineFinder();
    ~LineFinder();
    // �⺻ ���� �ػ󵵴� 1���� 1ȭ�� 
    // ������ ���� �ּ� ���̵� ����

    // �����⿡ �ػ� ����
    void setAccResolution(double dRho, double dTheta);

    // ��ǥ �ּ� ���� ����
    void setMinVote(int minv);

    // �� ���̿� ���� ����
    void setLineLengthAndGap(double length, double gap);

    // ���� �� ���׸�Ʈ ������ �����ϴ� �޼ҵ�
    // Ȯ���� ���� ��ȯ ����
    std::vector<cv::Vec4i> findLines(cv::Mat& binary);

    // �� �޼ҵ忡�� ������ ���� ���� �޼ҵ带 ����ؼ� �׸�
    // ���󿡼� ������ ���� �׸���
    void drawDetectedLines(cv::Mat &image, cv::Scalar color = cv::Scalar(255, 255, 255));
};