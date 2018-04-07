#include "opencv2/opencv.hpp" 
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>

#include "sdm_common.h"
//-------------params for training----------
const int NUM_ROUND = 5;
vector<HOGDescriptor> hogs(NUM_ROUND);
 
int num_lmks[NUM_ROUND] = { HOG_GLOBAL,15,22,22,HOG_ALL };
int id_lmks[NUM_ROUND][68] = { { },
  { 2,5,8,11,14, 31,33,35, 36, 39,42,45,56,57,58 },//15
  { 1, 3, 6, 8, 10, 13, 15, 18, 20, 23, 25, 30, 31, 35, 36, 39, 42, 45, 48, 51, 54, 57 },//22
  { 1, 3, 6, 8, 10, 13, 15, 18, 20, 23, 25, 30, 31, 35, 36, 39, 42, 45, 48, 51, 54, 57 },
  {} };

int normal_face_width = 128;   //normalize face detect rectangle 
int border = 64;    //normalize ROI size = normal_face_width + 2*border
string face_model = "haarcascade_frontalface_alt2.xml";
//--------------------------------------------
//in case fail to detect face, use this to init face Rect: 
//rect_x_y_w = M_eyes_to_rect * eyes_x1_x2_y
cv::Mat M_eyes_to_rect;
 
shape2d shape_mean;
vector<shape2d> train_shapes;
vector<cv::Mat> train_samples;

//L = M*R ---> L*RT = M*R*RT ---> L*RT*(lambda*I+R*RT)^-1 = M
cv::Mat solve_norm_equation(cv::Mat& L, cv::Mat& R)
{
  cv::Mat M;
  cv::Mat LRT = L*R.t(), RRT = R*R.t();

  cout << "Begin solving...";
  float lambda = 0.5;// 0.050f * ((float)cv::norm(RRT)) / (float)(RRT.rows);

  for (int i = 0; i < RRT.cols - 1; i++)
    RRT.at<float>(i, i) = lambda + RRT.at<float>(i, i);
  M = LRT*RRT.inv(cv::DECOMP_LU);
  cout << " done." << endl;
  return M;
}

//3dof face rect to 3dof eyes position
void train_map_facerect_and_eyes(vector<string>& vstrPts, vector<string>& vstrImg)
{
  int eyes_index[4] = { 36,39,42,45 };
  cv::CascadeClassifier face_cascade;
  if (!face_cascade.load(face_model))
  {
    cout << "Error loading face detection model." << endl;
    return;
  }

  cout << "Train mapping between eyes and detect-rectangle ....." << endl;
  int good = 0, numsamples = 200;
  cv::Mat L = Mat::zeros(3, 3, CV_32F);
  cv::Mat R = Mat::zeros(3, 3, CV_32F);
  cv::Mat M; //L=M*R
  for (int i = 0; i < numsamples/*vstrPts.size()*/; i++)
  {
    shape2d pts = read_pts_landmarks(vstrPts[i]);
    cv::Mat img = cv::imread(vstrImg[i], 0);

    float scale = 1.0;
    if (img.rows <= img.cols && img.rows > 200) scale = 160.0 / img.rows;
    if (img.rows > img.cols && img.cols > 200) scale = 160.0 / img.cols;

    cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_LINEAR);
    vector<cv::Rect> detected_faces;
    face_cascade.detectMultiScale(img, detected_faces, 1.2, 4, 0, cv::Size(50, 50));

    if (detected_faces.size() == 0)
      continue;

    cv::Point2f p0 = (pts[eyes_index[0]] + pts[eyes_index[1]])*scale*0.5;
    cv::Point2f p1 = (pts[eyes_index[2]] + pts[eyes_index[3]])*scale*0.5;

    cv::Rect rectface = detected_faces[0];
    if (rectface.contains(p0) && rectface.contains(p1))
    {
      good++;

      Mat ll = (Mat_<float>(3, 1) << rectface.x, rectface.y, (rectface.width + rectface.height) / 2);
      Mat rr = (Mat_<float>(3, 1) << p0.x, (p0.y + p1.y) / 2, p1.x);
      L = L + ll*rr.t();
      R = R + rr*rr.t();
    }
  }
  cout << "Collect " << good << " good faces in " << numsamples << " images. " << endl;

  M = L*R.inv();
  M_eyes_to_rect = M.clone();
  cout << "M_eyes_to_rect:\n" << M_eyes_to_rect << endl << endl;
}

void preprocess_images(vector<string>& vstrPts, vector<string>& vstrImg)
{
  int eyes_index[4] = { 36,39,42,45 };
  cv::CascadeClassifier face_cascade;
  face_cascade.load(face_model);

  bool showimg = true;
  Mat imcanvas;
  RNG rng((unsigned)time(NULL));
  cout << "Collecting cropped " << vstrPts.size() << " faces and normlized shapes ...";
  int count = 0;
  for (int i = 0; i < vstrPts.size(); i++)
  {
    shape2d pts = read_pts_landmarks(vstrPts[i]);
    cv::Mat img_ = cv::imread(vstrImg[i], 0);//-1 origimage,0 grayscale
    cv::Mat img;
    float scale = 1.0;
    if (img_.rows <= img_.cols && img_.rows > 500) scale = 400.0 / img_.rows;
    if (img_.rows > img_.cols && img_.cols > 500) scale = 400.0 / img_.cols;
    cv::resize(img_, img, cv::Size(), scale, scale, cv::INTER_LINEAR);

    vector<cv::Rect> detected_faces;
    face_cascade.detectMultiScale(img, detected_faces, 1.2, 4, 0, cv::Size(50, 50));

    cv::Point2f p0 = (pts[eyes_index[0]] + pts[eyes_index[1]])*scale*0.5;
    cv::Point2f p1 = (pts[eyes_index[2]] + pts[eyes_index[3]])*scale*0.5;

    int idx = -1;
    for (int i = 0; i < detected_faces.size(); i++)
    {
      //eye lmks in this face rectangle
      if (detected_faces[i].contains(p0) && detected_faces[i].contains(p1))
        idx = i;
    }

    cv::Rect rectface;
    //from eyes' position  to get face detection rectangle
    if (detected_faces.size() == 0 || idx == -1)
    { 
      Mat rr = (Mat_<float>(3, 1) << p0.x, (p0.y + p1.y) / 2, p1.x);
      Mat ll = M_eyes_to_rect*rr;
      rectface = cv::Rect(ll.at<float>(0), ll.at<float>(1), ll.at<float>(2), ll.at<float>(2));
      if (showimg)
      {
        cv::circle(img, Point(p0.x, p0.y), 2, cv::Scalar(255, 255, 255), -1);
        cv::circle(img, Point(p1.x, p1.y), 2, cv::Scalar(255, 255, 255), -1);
        cv::rectangle(img, rectface, cv::Scalar(255, 255, 255), 1);
      }
    }
    else
    {
      rectface = detected_faces[idx];
      if (showimg)
      {
        cv::rectangle(img, rectface, cv::Scalar(0, 0, 255), 1);

        //from face detection rect to get eyes' position        
        Mat ll = (Mat_<float>(3, 1) << rectface.x, rectface.y, rectface.width);
        Mat rr = M_eyes_to_rect.inv()*ll; //rr = M^-1*ll
        cv::circle(img, Point(rr.at<float>(0), rr.at<float>(1)), 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, Point(rr.at<float>(2), rr.at<float>(1)), 2, cv::Scalar(0, 0, 255), -1);
      }
    }
    if (showimg)
    {
      imshow("img", img);
      waitKey(100);
    }
    /////////perturb/////////// 
    float sigma = 0.15;
    float n1 = rng.gaussian(sigma), n2 = rng.gaussian(sigma), n3 = rng.gaussian(sigma);

    rectface.x += (n1-n3*0.5)*(float)rectface.width;
    rectface.y += (n2-n3*0.5)*(float)rectface.width;
    rectface.width +=  n3*(float)rectface.width ;
  //  cout << n1 << " " << n2 << " " << n3 << endl;

    cv::Mat roiImg;
    int normal_roi_width = normal_face_width + border * 2;

    float scale_again = (float)normal_face_width / (float)rectface.width;
    float shift_x = border - scale_again*rectface.x, shift_y = border - scale_again*rectface.y;
    Mat mat2roi = (Mat_<double>(2, 3) << scale_again, 0, shift_x, 0, scale_again, shift_y);
    warpAffine(img, roiImg, mat2roi, Size(normal_roi_width, normal_roi_width), INTER_LINEAR, BORDER_CONSTANT, 128);
    
    //lmks on normalized ROI
    float scale2roi = scale_again*scale;
    float dx = mat2roi.at<double>(0, 2);
    float dy = mat2roi.at<double>(1, 2);
    for (int j = 0; j < pts.size(); j++)
    {
      pts[j].x = pts[j].x *scale2roi + dx;
      pts[j].y = pts[j].y *scale2roi + dy;
    }

    //average shape at normal scale
    if (i == 0)
    {
      shape_mean.resize(pts.size());
      for (int j = 0; j < pts.size(); j++)
        shape_mean[j] = cv::Point2f(pts[j].x, pts[j].y);
    }
    else
    {
      for (int j = 0; j < pts.size(); j++)
        shape_mean[j] = shape_mean[j] + pts[j];
    }
    count++;

    train_shapes.push_back(pts);
    train_samples.push_back(roiImg);

    if (showimg)
    {
      cvtColor(roiImg, imcanvas, COLOR_GRAY2RGB);
      HOGDescriptor hog = hogs[3];
      int half_wid = hog.blockSize.width >> 1;
      for (int j = 0; j < pts.size(); j++)
      {
        if (pts[j].x<half_wid || pts[j].x + half_wid>roiImg.cols || pts[j].y + half_wid > roiImg.rows)
          cout << "Warning: HOG region out of image :" << i << " at lmk " << j << endl;
        cv::circle(imcanvas, cv::Point(pts[j].x, pts[j].y), 2, cv::Scalar(0, 0, 255), -1);
        Rect r(pts[j].x - half_wid, pts[j].y - half_wid, hog.blockSize.width, hog.blockSize.width);
        cv::rectangle(imcanvas, r, cv::Scalar(0, 255, 0), 1);
      }
      char name[200];
      sprintf(name, "output/f_%d.png", i);
      imwrite(name, imcanvas);
      imshow("Face ROI", imcanvas);
      waitKey(10);
    }
  }//end collecting

  for (int j = 0; j < shape_mean.size(); j++)    shape_mean[j] = shape_mean[j] * (1.0f / (float)count);
  cout << " done." << endl;
  if (showimg)
  {
    cv::Mat avgimg, temp;
    train_samples[0].convertTo(avgimg, CV_32FC1);

    for (int i = 1; i < train_samples.size(); i++)
    {
      train_samples[i].convertTo(temp, CV_32FC1);
      avgimg = avgimg + temp;
    }
    avgimg = avgimg / train_samples.size();

    HOGDescriptor hog = hogs[0]; //draw hog[0] feature rect
    int half_wid = hog.blockSize.width >> 1;
    avgimg.convertTo(imcanvas, CV_8UC1);
    cvtColor(imcanvas, imcanvas, COLOR_GRAY2RGB);

    for (int j = 0; j < shape_mean.size(); j++)
    {
      cv::circle(imcanvas, cv::Point(shape_mean[j].x, shape_mean[j].y), 2, cv::Scalar(0, 0, 255), -1);
      Rect r(shape_mean[j].x - half_wid, shape_mean[j].y - half_wid, hog.blockSize.width, hog.blockSize.width);
      cv::rectangle(imcanvas, r, cv::Scalar(0, 255, 0), 1);
    }
    cout << "Press any to continue." << endl;
    imshow("meanface with shape_mean", imcanvas);
    waitKey();
  }
}

void set_idx_lmks(vector<int>& num_lmks, vector<vector<int> >& lmk_all)
{
  for (int i = 0; i < num_lmks.size(); i++)
  {
    if (num_lmks[i] == HOG_GLOBAL || num_lmks[i] == HOG_ALL)
    {
      vector<int> t;
      lmk_all.push_back(t);
    }
    else
    {
      vector<int> t(id_lmks[i], id_lmks[i] + num_lmks[i]);
      lmk_all.push_back(t);
    }
  }
}

void train_by_regress()
{
  int numsample_test = 10; //left some for test
  int numsample = train_shapes.size() - numsample_test;
  int numpts = shape_mean.size();

  cout << "Training SDM on " << numsample << " images, left " << numsample_test << " for testing ..." << endl;

  vector<int> vnum_lmks(num_lmks, num_lmks + NUM_ROUND);
  vector<vector<int> > vlmk_all;
  set_idx_lmks(vnum_lmks, vlmk_all);

  //set HOG size 
  set_HOGs(hogs, normal_face_width + border, vnum_lmks);
  
  for (int i = 0; i < hogs.size(); i++)
  {
    cout << "HOG Round: " << i << " winsize: " << hogs[i].winSize << ", length of descriptor: " << hogs[i].getDescriptorSize() << endl;
    if (num_lmks[i] > 0) for (int j = 0; j < vnum_lmks[i]; j++)  cout << id_lmks[i][j] << ",";
    else { num_lmks[i] == 0 ? cout << "  use all 68 lmks" : cout << "  use whole face ROI"; };  cout << endl;
  }
  cout << endl;

  vector<shape2d> shape_current_all;
  //init as meashape
  for (int i = 0; i < numsample; i++)
    shape_current_all.push_back(shape_mean);

  Mat mDiffShape(numpts * 2, numsample, CV_32F);
  Mat mFeature;
  vector<cv::Mat> Mreg;
  for (int r = 0; r < hogs.size(); r++)
  {
    cout << "---- Round " << r << " -----" << endl;
    for (int i = 0; i < numsample; i++)
    {
      Mat roiImg = train_samples[i];
      vector<float>  des = extract_HOG_descriptor(roiImg, shape_current_all[i], hogs[r], vnum_lmks[r], vlmk_all[r]);
      if (i == 0)
      {
        mFeature.create(des.size(), numsample, CV_32F);
        cout << "   training sample: " << numsample << ", length of feature: " << des.size() << endl;
      }

      Mat rr = Mat(des.size(), 1, CV_32F, des.data());
      rr.copyTo(mFeature.col(i)); //must use copyTo

      Mat v_dest = shape2d_to_mat(train_shapes[i]) - shape2d_to_mat(shape_current_all[i]);
      v_dest.copyTo(mDiffShape.col(i));
    }//end collect faces   

    //regression
    cv::Mat M = solve_norm_equation(mDiffShape, mFeature);
    Mreg.push_back(M);

    //renew current shapes
    Mat mIncShape = M*mFeature;
    float E = 0.0f;
    for (int i = 0; i < numsample; i++)
    {
      Mat v_new = mIncShape.col(i) + shape2d_to_mat(shape_current_all[i]);
      shape_current_all[i] = mat_to_shape2d(v_new);

      Mat v_err = shape2d_to_mat(train_shapes[i]) - v_new;
      E = E + cv::norm(v_err, NORM_L2);
    }
    E /= numsample;
    cout << "  Avg Shape Error = " << E << endl;
  }//end training round  

  cout << "Save training result ...";
  string name_M = "face_sdm.bin";
  FileStorage fs("face_sdm.yml", FileStorage::WRITE);
  if (fs.isOpened())
  {
    fs << "NUM_LMKS" << numpts;
    fs << "TRAIN_ROUND" << NUM_ROUND;
    fs << "LMKS_EACH_ROUND" << vnum_lmks;
    for (int i = 0; i < vlmk_all.size(); i++)
    {
      string id_lmk = "ID_IN_ROUND_";
      //if (vnum_lmks[i] != -1 && vnum_lmks[i] != 0)
      // fs << id_lmk + to_string(_Longlong(i)) << vlmk_all[i];
       fs << id_lmk + to_string(long(i)) << vlmk_all[i];
    }

    fs << "NORMAL_WID" << normal_face_width << "NORMAL_BORDER" << border;
    fs << "OpenCV_HAAR" << face_model;
    fs << "SDM_Mat" << name_M;
    fs << "Mean_Shape" << shape_mean;
    fs.release();
  }
  mats_write(name_M.c_str(), Mreg);
  cout << " done." << endl;

  if (1)
  {
    cout << "Testing on unseen images ... ";
    Mat imcanvas;
    for (int i = numsample; i < train_samples.size(); i++)//on rest image in trainset
    {
      Mat roiImg = train_samples[i];
      cvtColor(roiImg, imcanvas, COLOR_GRAY2RGB);
      draw_shape(imcanvas, shape_mean, cv::Scalar(255, 0, 0));
      shape2d pts = shape_mean;
      for (int r = 0; r < Mreg.size(); r++)
      {
        vector<float>  des = extract_HOG_descriptor(roiImg, pts, hogs[r], vnum_lmks[r], vlmk_all[r]);
        Mat rr = Mat(des.size(), 1, CV_32F, des.data());
        Mat v_shape = Mreg[r] * rr + shape2d_to_mat(pts);
        pts = mat_to_shape2d(v_shape);

        if (r == 0) draw_shape(imcanvas, pts, cv::Scalar(0, 0, 255));
      }
      draw_shape(imcanvas, pts, cv::Scalar(0, 255, 0));
      char name[200];
      sprintf(name, "output/test_%d.png", i);
      imwrite(name, imcanvas);
    }
    cout << "done\n";
  }
}

int main()
{
  string imagepath = "data/helen/trainset";
  string ptlistname = imagepath + "/ptlist.txt"; //dir *.pts/b > ptlist.txt
  vector<string> vstrPts, vstrImg;

  read_names_pair(ptlistname, imagepath, string(), ".jpg", vstrPts, vstrImg);
  if(vstrPts.size()<30)
  {
    cout<<"Need more face images in path " << imagepath << endl;
    return 1;
  }

  train_map_facerect_and_eyes(vstrPts, vstrImg);  

  preprocess_images(vstrPts, vstrImg);
  train_by_regress();
  return 0;
}