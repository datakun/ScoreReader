#ifndef SCOREREADER_H
#define SCOREREADER_H

#include <QtWidgets/QMainWindow>
#include <QFileDialog>
#include <QImage>

#include "ui_scorereader.h"

#include <opencv/ml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "LineFinder.h"

#define PI 3.1415926

#define NOISE_SIZE 10

static const int MAX_K = 10;
static const int MAX_CLASS = 3;	// 클래스 수

static struct sSampleParam {
    int no_sample;				// 셈플 데이터의 수
    double mean_x, mean_y;		// x, y의 평균
    double stdev_x, stdev_y;	// x, y의 표준 편차
    CvScalar color_pt;			// 셈플 색
    CvScalar color_bg;			// 배경 색
} sample_param[MAX_CLASS] = {
    { 500, 500, 200, 60, 30, CV_RGB(180, 0, 0), CV_RGB(255, 0, 0), },
    { 1500, 200, 500, 100, 80, CV_RGB(0, 180, 0), CV_RGB(0, 255, 0), },
    { 1000, 400, 700, 60, 100, CV_RGB(0, 0, 180), CV_RGB(0, 0, 255), },
};

struct _lineArea {
    QRect rectArea;
    int lineY[5];
};
typedef _lineArea LineArea;

class ScoreReader : public QMainWindow
{
    Q_OBJECT

public:
    ScoreReader(QWidget *parent = 0);
    ~ScoreReader();

private:
    QPixmap cvMatToQPixmap(const cv::Mat &inMat);
    QImage cvMatToQImage(const cv::Mat &inMat);
    cv::Mat QImageToCvMat(const QImage &inImage, bool inCloneImageData = true);
    cv::Mat QPixmapToCvMat(const QPixmap &inPixmap, bool inCloneImageData = true);

    // 이미지 세선화
    void thinning(cv::Mat inMat);
    void thinningIteration(cv::Mat& inMat, int iter);

    // 허프 변환으로 가로선 찾기
    void findLineByHough(cv::Mat inMat);

    // 침식 팽창으로 가로선을 찾은 뒤, 원본 이미지에서 가로선 제거
    cv::Mat findLines(cv::Mat inMat, cv::Mat outLabeledMat, cv::Mat outLabelStats);
    cv::Mat removeLines(cv::Mat inMat, cv::Mat lineMat);

    // 가로선 제거된 이미지에서 객체 찾기
    // @param inMat the source image
    // @param outLabeledMat destination labeled image
    // @param outLabelStats destination labeled image statistics
    cv::Mat findObjects(cv::Mat inMat, cv::Mat outLabeledMat, cv::Mat outLabelStats);

    // 가로선 제거된 이미지에서 마디 찾기
    void findBar(cv::Mat inMat);

    void setDisplayOriginImage(const QImage image);
    void setDisplayOriginImage(const cv::Mat& inMat);
    void setDisplayResultImage(const QImage image);
    void setDisplayResultImage(const cv::Mat& inMat);

    void releaseImage(QImage *image);
    void releaseImage(QPixmap *pixmap);

    void KNN();

private:
    Ui::ScoreReaderClass ui;

    QFileDialog* m_fileDialog;

    QString m_fileName;
    QImage* m_scoreImage;
    QImage* m_scoreLineImage;
    QImage* m_resultImage;

    int m_scoreBar;
    QString m_scoreBeat;
    QString m_scoreKey;
    int m_scoreTempo;

    // 오선 영역 리스트
    QList<LineArea> m_lineAreaList;
};

#endif // SCOREREADER_H
