#pragma once
#include "opencv2/opencv.hpp" 
#include <vector>
#include <iostream>
#include <fstream>

#include "sdm_common.h"

enum { SDM_OK, LOAD_MODEL_FAIL, INPUT_FAIL }; 

struct SDM
{
  int numpts;
  int NUM_ROUND;
  int normal_face_width;
  int border;
  int min_face_size;
  
  shape2d shape_mean;
  
  vector<int> vnum_lmks;
  
  vector<vector<int> > vlmk_all;
  
  vector<HOGDescriptor> hogs;
  
  vector<Mat> Mreg;
  
  cv::CascadeClassifier face_cascade;
  
  vector<cv::Rect> detected_faces;

  SDM() { min_face_size = 50; };
  
  void set_min_face(int sz)
  {
    min_face_size = sz; 
  }
  
  bool init(string& configfile);  

  void regression(Mat &roiImg, shape2d &pts);   

  bool fit(Mat & img_, vector<shape2d>& shapes);  
};

