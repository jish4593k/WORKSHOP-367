import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.global.opencv_imgproc.*;

import javax.swing.*;

public class ShapeDetection {
    public static void main(String[] args) {
        Mat img = imread("resources/shapes.png");
        Mat imgContour = img.clone();

        Mat imgGray = new Mat();
        cvtColor(img, imgGray, COLOR_BGR2GRAY);
        Mat imgBlur = new Mat();
        GaussianBlur(imgGray, imgBlur, new Size(7, 7), 1);
        Mat imgCanny = new Mat();
        Canny(imgBlur, imgCanny, 50, 50);
        Mat imgBlank = new Mat(new Size(img.cols(), img.rows()), CV_8U, new Scalar(0));

        MatVector contours = new MatVector();
        findContours(imgCanny, contours, new Mat(), RETR_EXTERNAL, CHAIN_APPROX_NONE);

        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours.get(i));
            System.out.println(area);
            if (area > 500) {
                drawContours(imgContour, contours, i, new Scalar(255, 0, 0, 0), 3, LINE_8, new Mat(), 0, new Point());
            }
        }

        Mat imgStack = stackImages(0.6, new Mat[] {
                img, imgGray, imgBlur,
                imgCanny, imgContour, imgBlank
        });

        JFrame frame = new JFrame("Stacked Images");
        JLabel label = new JLabel(new ImageIcon(toBufferedImage(imgStack)));
        frame.add(label);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);
    }

    public static Mat stackImages(double scale, Mat[] imgArray) {
        int rows = imgArray.length;
        int cols = imgArray[0].cols();
        int type = imgArray[0].type();
        int height = imgArray[0].rows();

        for (Mat anImgArray : imgArray) {
            if (anImgArray.cols() != cols || anImgArray.type() != type || anImgArray.rows() != height) {
                throw new IllegalArgumentException("All images must have the same dimensions and type");
            }
        }

        Mat imageBlank = new Mat(new Size(cols, height), type, new Scalar(0));

        Mat[] hor = new Mat[rows];
        for (int i = 0; i < rows; i++) {
            hor[i] = new Mat();
            if (imgArray[i].type() == CV_8U) {
                cvtColor(imgArray[i], imgArray[i], COLOR_GRAY2BGR);
            }
            resize(imgArray[i], imgArray[i], new Size(), scale, scale, INTER_LINEAR);
        }

        vconcat(hor, imageBlank);

        return imageBlank;
    }

    public static BufferedImage toBufferedImage(Mat matImage) {
        int bufferSize = matImage.channels() * matImage.cols() * matImage.rows();
        byte[] bytes = new byte[bufferSize];
        matImage.get(0, 0, bytes);
        BufferedImage bufferedImage = new BufferedImage(matImage.cols(), matImage.rows(), BufferedImage.TYPE_BYTE_GRAY);
        bufferedImage.getRaster().setDataElements(0, 0, matImage.cols(), matImage.rows(), bytes);
        return bufferedImage;
    }
}
