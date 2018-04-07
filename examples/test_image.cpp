#include "opencv2/opencv.hpp" 
#include <vector>
#include <iostream>
#include <fstream>
 
#include "sdm_fit.h" 

int test_image(cv::Mat & img)
{
  SDM sdm_face;
  sdm_face.set_min_face(30);
  string config_name = "face_sdm.yml";
  if (sdm_face.init(config_name))
    cout << "Load SDM model\n";
  else
    return LOAD_MODEL_FAIL;

  float scale = 1.0;
  if (img.rows <= img.cols && img.rows > 500) scale = 480.0 / img.rows;
  if (img.rows > img.cols && img.cols > 500) scale = 480.0 / img.cols;

  if (scale < 1.0)
    cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_LINEAR);

  vector<shape2d> shapes;

  if (sdm_face.fit(img, shapes))
  {
    cout << shapes.size() << " faces found.\n";
    for (int j = 0; j < shapes.size(); j++)
    {      
      cv::rectangle(img, sdm_face.detected_faces[j], cv::Scalar(0,0,255));
      draw_shape(img, shapes[j], cv::Scalar(255, 0, 0));
    }
    imshow("image", img);
    waitKey();
  }
  return SDM_OK;
}
 

int main()
{
  int ret = -1;

  cv::Mat im = imread("data/helen/testset/325564774_1.jpg");
  ret = test_image(im);

  return ret;
}