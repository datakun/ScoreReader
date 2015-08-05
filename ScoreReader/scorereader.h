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

#define NOISE_SIZE 10

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

    // �̹��� ����ȭ
    void thinning(cv::Mat inMat);
    void thinningIteration(cv::Mat& inMat, int iter);

    // ���� ��ȯ���� ���μ� ã��
    void findLineByHough(cv::Mat inMat);

    // ħ�� ��â���� ���μ��� ã�� ��, ���� �̹������� ���μ� ����
    cv::Mat findLines(cv::Mat inMat, cv::Mat outLabeledMat, cv::Mat outLabelStats);
    cv::Mat removeLines(cv::Mat inMat, cv::Mat lineMat);

    // ���μ� ���ŵ� �̹������� ��ü ã��
    // @param inMat the source image
    // @param outLabeledMat destination labeled image
    // @param outLabelStats destination labeled image statistics
    cv::Mat findObjects(cv::Mat inMat, cv::Mat outLabeledMat, cv::Mat outLabelStats);

    // ���μ� ���ŵ� �̹������� ���� ã��
    void findBar(cv::Mat inMat);

    void setDisplayOriginImage(const QImage image);
    void setDisplayOriginImage(const cv::Mat& inMat);
    void setDisplayResultImage(const QImage image);
    void setDisplayResultImage(const cv::Mat& inMat);

    void releaseImage(QImage *image);
    void releaseImage(QPixmap *pixmap);

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

};

#endif // SCOREREADER_H
