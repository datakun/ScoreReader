#ifndef SCOREREADER_H
#define SCOREREADER_H

#include <QtWidgets/QMainWindow>
#include <QFileDialog>
#include <QImage>

#include "ui_scorereader.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "LineFinder.h"

#define PI 3.1415926

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
    void removeLines(cv::Mat inMat);

    // 가로선 제거된 이미지에서 객체 찾기
    void findObjects(cv::Mat inMat);

    // 가로선 제거된 이미지에서 마디 찾기
    void findBar(cv::Mat inMat);

    void setDisplayOriginImage(const QImage image);
    void setDisplayOriginImage(const cv::Mat& inMat);
    void setDisplayResultImage(const QImage image);
    void setDisplayResultImage(const cv::Mat& inMat);

private:
    Ui::ScoreReaderClass ui;

    QFileDialog* m_fileDialog;

    QString m_fileName;
    QImage* m_scoreImage;

    int m_scoreBar;
    QString m_scoreBeat;
    QString m_scoreKey;
    int m_scoreTempo;

};

#endif // SCOREREADER_H
