#include "opencv2/opencv.hpp" 
#include <vector>
#include <iostream>
#include <fstream>

#include "sdm_fit.h"
 
int test_webcam()
{
  SDM sdm_face;
  sdm_face.set_min_face(80);
  string config_name = "face_sdm.yml";
  if (sdm_face.init(config_name))  
    cout << "Load SDM model\n";
  else
    return LOAD_MODEL_FAIL;

  cv::Mat img, gray;

  VideoCapture capture;
  capture.open(0);

  if (capture.isOpened())
  {
    std::cout << "Capture is opened." << std::endl;
    for (;;)
    {
      capture >> img;
      if (img.empty())
      {
        cout << " capture error.\n";
        continue;
      }
      // resize(img, img, Size(), 0.5, 0.5, CV_INTER_LINEAR);
      cvtColor(img, gray, CV_BGR2GRAY);
      vector<shape2d> shapes;
      TIC;
      if (sdm_face.fit(gray, shapes))
      {
        //sdm_face.show_crop();
        //cout << shapes.size() << " faces found.\n";
        for (int j = 0; j < shapes.size(); j++)
        {
          cv::rectangle(img, sdm_face.detected_faces[j], cv::Scalar(0, 0, 255));
          draw_shape(img, shapes[j], cv::Scalar(255, 0, 0));
        }
      }
      TOC;
      imshow("Video", img);
      if (waitKey(10) == 27)
        break;
    }
  }
  else
  {
    cout << " NO camera input." << endl;
    return INPUT_FAIL;
  }
  return SDM_OK;
}

int main()
{
  int ret   = test_webcam(); 

  return ret;
}