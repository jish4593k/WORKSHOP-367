import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_videoio.*;

public class MotionDetection {
    public static void main(String[] args) {
        VideoCapture cam = new VideoCapture(0);
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        Mat firstFrame = null;
        int area = 700;

        while (true) {
            Mat img = new Mat();
            cam.read(img);
            String text = "Normal";

            Imgproc.resize(img, img, new Size(500, 500));
            Mat grayImg = new Mat();
            Imgproc.cvtColor(img, grayImg, Imgproc.COLOR_BGR2GRAY);
            Mat gaussianImg = new Mat();
            Imgproc.GaussianBlur(grayImg, gaussianImg, new Size(31, 31), 0);

            if (firstFrame == null) {
                firstFrame = gaussianImg.clone();
                continue;
            }

            Mat imgDiff = new Mat();
            Core.absdiff(firstFrame, gaussianImg, imgDiff);
            Mat threshImg = new Mat();
            Imgproc.threshold(imgDiff, threshImg, 25, 255, Imgproc.THRESH_BINARY);
            Imgproc.dilate(threshImg, threshImg, new Mat(), new Point(-1, -1), 2);

            MatVector contours = new MatVector();
            Imgproc.findContours(threshImg.clone(), contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            for (int i = 0; i < contours.size(); i++) {
                double areaValue = Imgproc.contourArea(contours.get(i));
                if (areaValue < area) {
                    continue;
                }

                Rect rect = Imgproc.boundingRect(contours.get(i));
                Imgproc.rectangle(img, new Point(rect.x(), rect.y()), new Point(rect.x() + rect.width(), rect.y() + rect.height()), new Scalar(0, 255, 0), 2);
                text = "Moving Object Detected";
            }

            System.out.println(text);
            Imgproc.putText(img, text, new Point(10, 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 0, 255), 2);

            Imgcodecs.imwrite("CameraFeed.jpg", img);
            HighGui.imshow("CameraFeed", img);

            int key = HighGui.waitKey(1);
            if (key == 'q' || key == 27) {
                break;
            }
        }

        cam.release();
        HighGui.destroyAllWindows();
    }
}
