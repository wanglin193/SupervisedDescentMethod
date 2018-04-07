#include "opencv2/opencv.hpp" 
#include <vector>
#include <iostream>
#include <fstream>
 
#include "sdm_fit.h" 

bool SDM::init(string& configfile)
{
  string name_M, face_model;// = "haarcascade_frontalface_alt2.xml";
  cv::FileStorage fs(configfile, cv::FileStorage::READ);
  if (!fs.isOpened()) { cout << "ERROR: Wrong path to settings" << endl;  return false; }
  fs["NUM_LMKS"] >> numpts;
  fs["TRAIN_ROUND"] >> NUM_ROUND;
  fs["LMKS_EACH_ROUND"] >> vnum_lmks;
  fs["NUM_LMKS"] >> numpts;
  for (int i = 0; i < vnum_lmks.size(); i++)
  {
    vector<int> obj;
    //fs["ID_IN_ROUND_" + to_string(_Longlong(i))] >> obj;
    fs["ID_IN_ROUND_" + to_string(long(i))] >> obj;
    vlmk_all.push_back(obj);
  }
  fs["NORMAL_WID"] >> normal_face_width;
  fs["NORMAL_BORDER"] >> border;
  fs["OpenCV_HAAR"] >> face_model;

  fs["Mean_Shape"] >> shape_mean;
  fs["SDM_Mat"] >> name_M;
  face_cascade.load(face_model);

  Mreg = mats_read(name_M.c_str());
  hogs.resize(NUM_ROUND);
 
  //set HOG size 
  set_HOGs(hogs, normal_face_width + border, vnum_lmks);
  return true;
}

void SDM::regression(Mat &roiImg, shape2d &pts)
{
  for (int r = 0; r < Mreg.size(); r++)
  {
    vector<float>  des = extract_HOG_descriptor(roiImg, pts, hogs[r], vnum_lmks[r], vlmk_all[r]);
    Mat rr = Mat(des.size(), 1, CV_32F, des.data());
    Mat v_shape = Mreg[r] * rr + shape2d_to_mat(pts);
    pts = mat_to_shape2d(v_shape);
  }
}

bool SDM::fit(Mat & img_, vector<shape2d>& shapes)
{
  cv::Mat img, img_gray;

  if (img_.channels() > 1)
    cvtColor(img_, img_gray, CV_BGR2GRAY);
  else
    img_gray = img_.clone();
  
  face_cascade.detectMultiScale(img_gray, detected_faces, 1.2, 4, 0, cv::Size(min_face_size, min_face_size));
  
  if (detected_faces.size() == 0)
    return false;

  shapes.clear();
  for (int i = 0; i < detected_faces.size(); i++)
  {   
    cv::Rect rectface = detected_faces[i];
    cv::Mat roiImg;
    int normal_roi_width = normal_face_width + border * 2;  
    
    float scale = (float)normal_face_width / (float)rectface.width;
    float shift_x = border - scale*rectface.x, shift_y = border - scale*rectface.y;
    Mat mat2roi = (Mat_<double>(2, 3) << scale, 0, shift_x, 0, scale, shift_y);
    warpAffine(img_gray, roiImg, mat2roi, Size(normal_roi_width, normal_roi_width), INTER_LINEAR, BORDER_CONSTANT, 128);
  
    shape2d pts(shape_mean.size());
    if (1)//detect mode: start from meanshape
    {
      for (int j = 0; j < shape_mean.size(); j++) 
        pts[j] = shape_mean[j];
    }
    //else //tracking mode: update shape from current state (or previous frame)
    //{
    //  for (int j = 0; j < pts.size(); j++)
    //  {
    //    pts[j].x =  
    //    pts[j].y =  
    //  }
    //}
    regression(roiImg, pts);
    if (1)
    {
      string win_name = "roi ";
      cv::Rect roi(border, border, normal_face_width, normal_face_width);
      cv::rectangle(roiImg, roi, cv::Scalar(255));
      draw_shape(roiImg, pts, cv::Scalar(255));
     // imshow(win_name + to_string(_Longlong(i)), roiImg);
       imshow(win_name + to_string(long(i)), roiImg);
    }
    //back to original img  
    float scale_back = 1.0 / mat2roi.at<double>(0, 0);
    float dx_back = -scale_back*mat2roi.at<double>(0, 2);
    float dy_back = -scale_back*mat2roi.at<double>(1, 2);
    for (int j = 0; j < pts.size(); j++)
    {
      pts[j].x = pts[j].x*scale_back + dx_back;
      pts[j].y = pts[j].y*scale_back + dy_back;
    }
    shapes.push_back(pts); 
  }
  return true;
}
  