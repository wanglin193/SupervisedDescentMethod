#include "opencv2/opencv.hpp" 
#include <vector>
#include <iostream>
#include <fstream>
 
#include "sdm_fit.h" 
 
int test_on_testset()
{
  SDM sdm_face;
  string config_name = "face_sdm.yml";
  if (sdm_face.init(config_name))  
    cout << "Load SDM model\n";
  else
    return LOAD_MODEL_FAIL;

  string imagepath = "data/helen/testset";
  //dir *.pts/b>ptlist.txt
  //ls *.pts -1>ptlist.txt
  string ptlistname = imagepath + "/ptlist.txt"; 
  vector<string> vstrPts, vstrImg;

  read_names_pair(ptlistname, imagepath, string(), ".jpg", vstrPts, vstrImg);
  if (vstrImg.empty())
    return INPUT_FAIL;

  for (int i = 0; i < vstrImg.size(); i++)
  {
    cv::Mat img = cv::imread(vstrImg[i], -1);//-1 origimage,0 grayscale
    float scale = 1.0;
    if (img.rows <= img.cols && img.rows > 400) scale = 320.0 / img.rows;
    if (img.rows > img.cols && img.cols > 400) scale = 320.0 / img.cols;

    if (scale < 1.0)
      cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_LINEAR);

    cout << vstrImg[i] << endl;
    vector<shape2d> shapes;

    TIC;
    int ret = sdm_face.fit(img, shapes);
    TOC;
    if (ret)
    {
      cout << shapes.size() << " faces found.\n";
      for (int j = 0; j < shapes.size(); j++)
        draw_shape(img, shapes[j], cv::Scalar(255, 0, 0));

      imshow("faces", img);
      waitKey(100);

      //find name without path
      int pos1=vstrImg[i].find_last_of('/');
      string file_name(vstrImg[i].substr(pos1+1));
      imwrite("output/" + file_name, img);
    }
    else
      cout << "No face found\n";
  }
  return SDM_OK;
}
 

int main()
{
  int ret =   test_on_testset();

  return ret;
}