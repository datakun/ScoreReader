#include "scorereader.h"

#include "qdebug.h"

#ifdef _DEBUG
#pragma comment(lib, "opencv_core300d.lib")  
#pragma comment(lib, "opencv_highgui300d.lib")
#pragma comment(lib, "opencv_imgproc300d.lib")
#pragma comment(lib, "opencv_imgcodecs300d.lib")
#else
#pragma comment(lib, "opencv_core300.lib")  
#pragma comment(lib, "opencv_highgui300.lib")
#pragma comment(lib, "opencv_imgproc300.lib")
#pragma comment(lib, "opencv_imgcodecs300.lib")
#endif

ScoreReader::ScoreReader(QWidget *parent)
    : QMainWindow(parent), m_fileDialog(NULL), m_scoreImage(NULL)
{
    ui.setupUi(this);

    connect(ui.actionClose, &QAction::triggered, this, &QMainWindow::close);
    connect(ui.actionOpen, &QAction::triggered, [=]{
        m_fileDialog = new QFileDialog(this, tr("Open Image"), "", tr("Image Files (*.png *.jpg *.bmp *.gif)"));
        if (m_fileDialog->exec())
        {
            m_fileName = m_fileDialog->selectedFiles()[0];

            if (m_scoreImage)
                delete m_scoreImage;
            m_scoreImage = new QImage(m_fileName);
            ui.originLabel->setPixmap(QPixmap::fromImage(*m_scoreImage));

            setDisplayOriginImage(*m_scoreImage);
        }
    });
    connect(ui.actionFineLine, &QAction::triggered, [=]{
        cv::Mat srcMat = QImageToCvMat(m_scoreImage->convertToFormat(QImage::Format_RGB32));
        if (srcMat.empty())
            return;

        // TODO: �̹��� ó���� GRAY2BGR �ʿ�
        cv::Mat resultMat;
        cv::cvtColor(srcMat, resultMat, CV_BGR2GRAY);
        cv::threshold(resultMat, resultMat, 128, 255, CV_THRESH_BINARY_INV);

        //// ����ȭ
        //thinning(resultMat);

        ////setDisplayImage(resultMat);

        //// ������ȯ���� �� ã��
        ////findLineByHough(resultMat);

        removeLines(resultMat);

        //setDisplayResultImage(resultMat);

        findObjects(resultMat);

        setDisplayResultImage(resultMat);
    });
}

ScoreReader::~ScoreReader()
{
}

void ScoreReader::removeLines(cv::Mat inMat)
{
    cv::Mat horizontal = inMat.clone();

    int horizontalsize = horizontal.cols / 30;

    cv::Mat horizontalStructure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(horizontalsize, 1));

    cv::erode(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));
    cv::dilate(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));

    cv::absdiff(inMat, horizontal, horizontal);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 2));

    cv::dilate(horizontal, horizontal, kernel, cv::Point(-1, -1));
    cv::erode(horizontal, horizontal, kernel, cv::Point(-1, -1));

    //horizontal.copyTo(inMat);

    bitwise_not(horizontal, inMat);
}

void ScoreReader::findBar(cv::Mat inMat)
{
    m_scoreBar = 0;
    m_scoreKey = "C";
}

void ScoreReader::findObjects(cv::Mat inMat)
{
    //qDebug() << cv::connectedComponentsWithStats(coppied, inMat, inMat, inMat, 8);

    cv::Mat labelImage(inMat.size(), CV_32S);

    int nLabels = connectedComponents(inMat, labelImage, 8);
    cv::Vec3b *colors = new cv::Vec3b[nLabels];
    colors[0] = cv::Vec3b(0, 0, 0);//background

    for (int label = 1; label < nLabels; ++label){
        colors[label] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }

    cv::Mat dst(inMat.size(), CV_8UC3);
    for (int r = 0; r < dst.rows; ++r){
        for (int c = 0; c < dst.cols; ++c){
            int label = labelImage.at<int>(r, c);
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
            pixel = colors[label];

            // TODO: Ư�� �ȼ� �� ����
            //inMat.at<cv::Vec3b>(cv::Point(r, c)) = pixel;
        }
    }

    //coppied.copyTo(inMat);
}

void ScoreReader::findLineByHough(cv::Mat inMat)
{
    LineFinder ld; // �ν��Ͻ� ����

    // Ȯ���� ������ȯ �Ķ���� �����ϱ�
    ld.setLineLengthAndGap(100, 5);
    ld.setMinVote(80);

    // ���� �����ϰ� �׸���
    std::vector<cv::Vec4i> li = ld.findLines(inMat);

    inMat.setTo(cv::Scalar(0, 0, 0));

    ld.drawDetectedLines(inMat);

    bitwise_not(inMat, inMat);
}

void ScoreReader::thinning(cv::Mat inMat)
{
    inMat /= 255;

    cv::Mat prev = cv::Mat::zeros(inMat.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(inMat, 0);
        thinningIteration(inMat, 1);
        cv::absdiff(inMat, prev, diff);
        inMat.copyTo(prev);
    } while (cv::countNonZero(diff) > 0);

    inMat *= 255;

    bitwise_not(inMat, inMat);
}

void ScoreReader::thinningIteration(cv::Mat& inMat, int iter)
{
    cv::Mat marker = cv::Mat::zeros(inMat.size(), CV_8UC1);

    for (int i = 1; i < inMat.rows - 1; i++)
    {
        for (int j = 1; j < inMat.cols - 1; j++)
        {
            uchar p2 = inMat.at<uchar>(i - 1, j);
            uchar p3 = inMat.at<uchar>(i - 1, j + 1);
            uchar p4 = inMat.at<uchar>(i, j + 1);
            uchar p5 = inMat.at<uchar>(i + 1, j + 1);
            uchar p6 = inMat.at<uchar>(i + 1, j);
            uchar p7 = inMat.at<uchar>(i + 1, j - 1);
            uchar p8 = inMat.at<uchar>(i, j - 1);
            uchar p9 = inMat.at<uchar>(i - 1, j - 1);

            int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i, j) = 1;
        }
    }

    inMat &= ~marker;
}

void ScoreReader::setDisplayOriginImage(const QImage image)
{
    ui.originLabel->setPixmap(QPixmap::fromImage(image));
}

void ScoreReader::setDisplayOriginImage(const cv::Mat& inMat)
{
    ui.originLabel->setPixmap(cvMatToQPixmap(inMat));
}

void ScoreReader::setDisplayResultImage(const QImage image)
{
    ui.resultLabel->setPixmap(QPixmap::fromImage(image));
}

void ScoreReader::setDisplayResultImage(const cv::Mat& inMat)
{
    ui.resultLabel->setPixmap(cvMatToQPixmap(inMat));
}

cv::Mat ScoreReader::QImageToCvMat(const QImage &inImage, bool inCloneImageData)
{
    switch (inImage.format())
    {
        // 8-bit, 4 channel
    case QImage::Format_RGB32:
    {
        cv::Mat  mat(inImage.height(), inImage.width(), CV_8UC4, const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine());

        return (inCloneImageData ? mat.clone() : mat);
    }

    // 8-bit, 3 channel
    case QImage::Format_RGB888:
    {
        if (!inCloneImageData)
            qWarning() << "ASM::QImageToCvMat() - Conversion requires cloning since we use a temporary QImage";

        QImage   swapped = inImage.rgbSwapped();

        return cv::Mat(swapped.height(), swapped.width(), CV_8UC3, const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine()).clone();
    }

    // 8-bit, 1 channel
    case QImage::Format_Indexed8:
    {
        cv::Mat  mat(inImage.height(), inImage.width(), CV_8UC1, const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine());

        return (inCloneImageData ? mat.clone() : mat);
    }

    default:
        qWarning() << "ASM::QImageToCvMat() - QImage format not handled in switch:" << inImage.format();
        break;
    }

    return cv::Mat();
}

cv::Mat ScoreReader::QPixmapToCvMat(const QPixmap &inPixmap, bool inCloneImageData)
{
    return QImageToCvMat(inPixmap.toImage(), inCloneImageData);
}

QImage ScoreReader::cvMatToQImage(const cv::Mat &inMat)
{
    switch (inMat.type())
    {
        // 8-bit, 4 channel
    case CV_8UC4:
    {
        QImage image(inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB32);

        return image;
    }

    // 8-bit, 3 channel
    case CV_8UC3:
    {
        cvtColor(inMat, inMat, CV_BGR2RGB);
        QImage image(inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB888);

        return image;
    }

    // 8-bit, 1 channel
    case CV_8UC1:
    {
        static QVector<QRgb>  sColorTable;

        // only create our color table once
        if (sColorTable.isEmpty())
        {
            for (int i = 0; i < 256; ++i)
                sColorTable.push_back(qRgb(i, i, i));
        }

        QImage image(inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_Indexed8);

        image.setColorTable(sColorTable);

        return image;
    }

    default:
        qDebug() << "cv::Mat image type not handled in switch:" << inMat.type();
        break;
    }

    return QImage();
}

QPixmap ScoreReader::cvMatToQPixmap(const cv::Mat &inMat)
{
    return QPixmap::fromImage(cvMatToQImage(inMat));
}