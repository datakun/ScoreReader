#include "scorereader.h"

#include "qdebug.h"

#ifdef _DEBUG
#pragma comment(lib, "opencv_core300d.lib")  
#pragma comment(lib, "opencv_highgui300d.lib")
#pragma comment(lib, "opencv_imgproc300d.lib")
#pragma comment(lib, "opencv_imgcodecs300d.lib")
#pragma comment(lib, "opencv_ml300d.lib")
#else
#pragma comment(lib, "opencv_core300.lib")  
#pragma comment(lib, "opencv_highgui300.lib")
#pragma comment(lib, "opencv_imgproc300.lib")
#pragma comment(lib, "opencv_imgcodecs300.lib")
#pragma comment(lib, "opencv_ml300.lib")
#endif

ScoreReader::ScoreReader(QWidget *parent)
    : QMainWindow(parent), m_fileDialog(NULL), m_scoreImage(NULL), m_resultImage(NULL), m_scoreLineImage(NULL)
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
        if (m_scoreImage == NULL)
            return;

        cv::Mat srcMat = QImageToCvMat(m_scoreImage->convertToFormat(QImage::Format_RGB32));
        if (srcMat.empty())
            return;

        cv::cvtColor(srcMat, srcMat, CV_BGR2GRAY);
        cv::threshold(srcMat, srcMat, 128, 255, CV_THRESH_BINARY_INV);

        //// 세선화
        //thinning(resultMat);

        ////setDisplayImage(resultMat);

        //// 허프변환으로 선 찾기
        ////findLineByHough(resultMat);

        // 오선 찾은 뒤 지우기
        cv::Mat labeledLine;
        cv::Mat labeledLineStats;
        cv::Mat lineMat = findLines(srcMat, labeledLine, labeledLineStats);
        releaseImage(m_scoreLineImage);
        m_scoreLineImage = new QImage(cvMatToQImage(lineMat));
        cv::Mat resultMat = removeLines(srcMat, lineMat);

        // 오선 지워진 영상에서 객체 라벨링
        cv::Mat labeledImage;
        cv::Mat labeledImageStats;
        cv::Mat outMat = findObjects(resultMat, labeledImage, labeledImageStats);
        releaseImage(m_resultImage);
        m_resultImage = new QImage(cvMatToQImage(outMat));



        setDisplayResultImage(outMat);
    });
}

ScoreReader::~ScoreReader()
{
    releaseImage(m_scoreImage);
    releaseImage(m_scoreLineImage);
    releaseImage(m_resultImage);
}

void ScoreReader::releaseImage(QImage *image)
{
    if (image)
        delete image;
    image = NULL;
}

void ScoreReader::releaseImage(QPixmap *pixmap)
{
    if (pixmap)
        delete pixmap;
    pixmap = NULL;
}

cv::Mat ScoreReader::findLines(cv::Mat inMat, cv::Mat outLabeledMat, cv::Mat outLabelStats)
{
    // 가로가 긴 마스크로 침식 팽창을 하여 오선 후보 찾음
    cv::Mat horizontal = inMat.clone();

    int horizontalsize = horizontal.cols / 30;

    cv::Mat horizontalStructure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(horizontalsize, 1));

    cv::erode(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));
    cv::dilate(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));

    outLabeledMat.create(inMat.size(), CV_32S);

    // 잡음을 제거하기 위한 라벨링
    int nLabels = connectedComponents(horizontal, outLabeledMat, 8);
    outLabelStats.create(cv::Size(cv::CC_STAT_MAX, nLabels), cv::DataType<int>::type);
    cv::Mat centroids;
    connectedComponentsWithStats(horizontal, outLabeledMat, outLabelStats, centroids, 8);

    // 잡음 제거
    for (int i = 0; i < outLabeledMat.rows; ++i)
    {
        for (int j = 0; j < outLabeledMat.cols; ++j)
        {
            int label = outLabeledMat.at<int>(i, j);

            // 객체(선)의 길이가 영상의 (가로 너비) / 2 보다 작다면 잡음으로 인식
            if (outLabelStats.at<int>(label, cv::CC_STAT_WIDTH) <= horizontal.cols / 2)
                horizontal.at<uchar>(i, j) = 0;
        }
    }

    // 잡음이 제거 된 영상에서 다시 라벨링
    nLabels = connectedComponents(horizontal, outLabeledMat, 8);
    outLabelStats.create(cv::Size(cv::CC_STAT_MAX, nLabels), cv::DataType<int>::type);
    connectedComponentsWithStats(horizontal, outLabeledMat, outLabelStats, centroids, 8);

    for (int label = 1; label < nLabels; ++label)
    {
        int objWidth = outLabelStats.at<int>(label, cv::CC_STAT_WIDTH);
        int objHeight = outLabelStats.at<int>(label, cv::CC_STAT_HEIGHT);
        int objArea = outLabelStats.at<int>(label, cv::CC_STAT_AREA);
        cv::Point objCenter = cv::Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1));

        // 오선 영역 찾기

        // 1, 2, 3, 4, 5 번째 줄의 y 위치 찾기
    }

    return horizontal;
}

cv::Mat ScoreReader::removeLines(cv::Mat inMat, cv::Mat lineMat)
{
    cv::absdiff(inMat, lineMat, inMat);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 2));

    cv::dilate(inMat, inMat, kernel, cv::Point(-1, -1));
    cv::erode(inMat, inMat, kernel, cv::Point(-1, -1));

    bitwise_not(inMat, inMat);

    return inMat.clone();
}

void ScoreReader::findBar(cv::Mat inMat)
{
    m_scoreBar = 0;
    m_scoreKey = "C";
}

cv::Mat ScoreReader::findObjects(cv::Mat inMat, cv::Mat outLabeledMat, cv::Mat outLabelStats)
{
    bitwise_not(inMat, inMat);

    outLabeledMat.create(inMat.size(), CV_32S);

    // 잡음을 제거하기 위한 라벨링
    int nLabels = connectedComponents(inMat, outLabeledMat, 8);
    outLabelStats.create(cv::Size(cv::CC_STAT_MAX, nLabels), cv::DataType<int>::type);
    cv::Mat centroids;
    connectedComponentsWithStats(inMat, outLabeledMat, outLabelStats, centroids, 8);

    for (int i = 0; i < outLabeledMat.rows; ++i)
    {
        for (int j = 0; j < outLabeledMat.cols; ++j)
        {
            int label = outLabeledMat.at<int>(i, j);

            int objWidth = outLabelStats.at<int>(label, cv::CC_STAT_WIDTH);
            int objHeight = outLabelStats.at<int>(label, cv::CC_STAT_HEIGHT);
            int objArea = outLabelStats.at<int>(label, cv::CC_STAT_AREA);

            if (objArea <= NOISE_SIZE || objWidth > objHeight * 2)
                inMat.at<uchar>(i, j) = 0;
        }
    }

    // 잡음이 제거 된 영상에서 다시 라벨링
    nLabels = connectedComponents(inMat, outLabeledMat, 8);
    outLabelStats.create(cv::Size(cv::CC_STAT_MAX, nLabels), cv::DataType<int>::type);
    connectedComponentsWithStats(inMat, outLabeledMat, outLabelStats, centroids, 8);

    cv::Vec3b *colors = new cv::Vec3b[nLabels];
    colors[0] = cv::Vec3b(0, 0, 0); // 배경색

    for (int label = 1; label < nLabels; ++label)
        colors[label] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));

    cv::Mat outMat(inMat.size(), CV_8UC3);
    for (int i = 0; i < outMat.rows; ++i)
    {
        for (int j = 0; j < outMat.cols; ++j)
        {
            int label = outLabeledMat.at<int>(i, j);
            cv::Vec3b &pixel = outMat.at<cv::Vec3b>(i, j);
            pixel = colors[label];
        }
    }

    return outMat;
}

void ScoreReader::KNN()
{
    //cv::RNG rng;

    //IplImage *img = cvCreateImage(cvSize(1000, 1000), IPL_DEPTH_8U, 3);
    //cvZero(img);

    //cvNamedWindow("result", CV_WINDOW_AUTOSIZE);

    //// 학습 데이터의 총 수를 카운트 한다.
    //int sample_count = 0;
    //for (int i = 0; i<MAX_CLASS; i++) sample_count += sample_param[i].no_sample;

    //// 학습 데이터와 클래스를 할당할 행렬 생성
    //CvMat *train_data = cvCreateMat(sample_count, 2, CV_32FC1);
    //CvMat *train_class = cvCreateMat(sample_count, 1, CV_32SC1);

    //// 각 클래스 별로 정규분포를 가지는 학습 데이터를 무작위로 생성
    //for (int i = 0, k = 0; i<MAX_CLASS; i++){
    //    for (int j = 0; j<sample_param[i].no_sample; j++){
    //        CV_MAT_ELEM(*train_data, float, k, 0) = (float)(sample_param[i].mean_x + rng.gaussian(sample_param[i].stdev_x));
    //        CV_MAT_ELEM(*train_data, float, k, 1) = (float)(sample_param[i].mean_y + rng.gaussian(sample_param[i].stdev_y));
    //        CV_MAT_ELEM(*train_class, long, k, 0) = i;
    //        k++;
    //    }
    //}

    //// learn classifier
    //CvKNearest knn(train_data, train_class, 0, false, MAX_K);

    //CvMat *nearests = cvCreateMat(1, MAX_K, CV_32FC1);

    //// KNN 분류기가 이미지의 모든 픽셀에 대해 각 픽셀이 
    //// 어느 클래스에 속하는지 추정하여 클래스를 할당한다.
    //for (int x = 0; x<img->width; x++) {
    //    for (int y = 0; y<img->height; y++) {
    //        float sample_[2] = { (float)x, (float)y };
    //        CvMat sample = cvMat(1, 2, CV_32FC1, sample_);

    //        // KNN 분류기가 주어진 픽셀이 어느 클래스에 속하는지 추정한다.
    //        float response = knn.find_nearest(&sample, MAX_K, 0, 0, nearests, 0);

    //        // 이미지에 추정된 클래스를 색으로 표시한다.
    //        cvSet2D(img, y, x, sample_param[cvRound(response)].color_pt);
    //    }
    //}

    //// 학습 데이터를 이미지에 그린다.
    //for (int k = 0; k<sample_count; k++) {
    //    int x = cvRound(CV_MAT_ELEM(*train_data, float, k, 0));
    //    int y = cvRound(CV_MAT_ELEM(*train_data, float, k, 1));
    //    int c = cvRound(CV_MAT_ELEM(*train_class, long, k, 0));

    //    cvCircle(img, cvPoint(x, y), 2, sample_param[c].color_bg, CV_FILLED);
    //}

    //cvShowImage("result", img);

    //// 키를 누르면 종료
    //cvWaitKey(0);

    //cvReleaseMat(&train_class);
    //cvReleaseMat(&train_data);
    //cvReleaseMat(&nearests);

    //cvDestroyWindow("result");
    //cvReleaseImage(&img);
}

void ScoreReader::findLineByHough(cv::Mat inMat)
{
    LineFinder ld; // 인스턴스 생성

    // 확률적 허프변환 파라미터 설정하기
    ld.setLineLengthAndGap(100, 5);
    ld.setMinVote(80);

    // 선을 감지하고 그리기
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

        QImage swapped = inImage.rgbSwapped();

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