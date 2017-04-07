#include "opencv2/opencv.hpp" 
#include <vector>
#include <iostream>
#include <fstream>
 
using namespace std;
using namespace cv;

vector<HOGDescriptor> hogs(5);
int eyes_index[4] = { 36,39,42,45 };
vector<int> vshortlmk = { 1, 3, 6, 8, 10, 13, 15, 18, 20, 23, 25, 36, 39, 42, 45, 30, 31, 35, 48, 51, 54, 57 };
string face_model = "haarcascade_frontalface_alt2.xml";
int normal_face_width = 128;   //normalize face detect rectangle 
int border = 32;    //images size = normal_face_width+2*border

//in case fail to detect face, use this to init face Rect: 
//rect_x_y_w = M_eyes_to_rect * eyes_x1_x2_y
cv::Mat M_eyes_to_rect;
using shape2d = vector<cv::Point2f>;
shape2d shape_mean;
vector<shape2d> train_shapes;
vector<cv::Mat> train_samples;

inline void set_hog_params(HOGDescriptor&hog, int pix_in_cell, int cell_in_block, int n_orient)
{
   hog.cellSize = Size(pix_in_cell, pix_in_cell);
   hog.blockSize = cv::Size(cell_in_block * hog.cellSize.width, cell_in_block * hog.cellSize.height);
   hog.winSize = hog.blockSize; //this define feature region
   hog.blockStride = hog.cellSize;//useless when set hog.winSize = hog.blockSize; 
   hog.nbins = n_orient;
}
void set_HOGs()
{ 
  set_hog_params(hogs[0], (int)(normal_face_width*0.1 + 0.5), 4, 6);
  set_hog_params(hogs[1], (int)(normal_face_width*0.1 + 0.5), 4, 4);
  set_hog_params(hogs[2], (int)(normal_face_width*0.1 + 0.5), 4, 4);
  set_hog_params(hogs[3], (int)(normal_face_width*0.1 + 0.5), 4, 4);
  set_hog_params(hogs[4], (int)(normal_face_width*0.1 + 0.5), 3, 4);
  //for(auto hog:hogs)    cout << hog.winSize<< hog.getDescriptorSize()<< endl; 
}
//L = M*R ---> L*RT = M*R*RT ---> L*RT*(lambda*I+R*RT)^-1 = M
cv::Mat solve_norm_equation(cv::Mat& L, cv::Mat& R)
{
  cv::Mat M;
  cv::Mat LRT = L*R.t();
  cv::Mat RRT = R*R.t();

  cout << "Begin solving...";
  float lambda = 0.050f * (cv::norm(RRT)) / (float)(RRT.rows);
  // cout << "lambda " << lambda << endl;

  for (int i = 0; i < RRT.cols - 1; i++)
    RRT.at<float>(i, i) = lambda + RRT.at<float>(i, i);
  M = LRT*RRT.inv(cv::DECOMP_LU);  
  cout << " done." << endl;

  cv::Mat  res = L - M*R;
  //cout << "error " << cv::norm(res, cv::NORM_L2) << endl;
  return M;
}
cv::Mat solve_sgd(cv::Mat& L, cv::Mat& R)
{
  cv::Mat M = Mat::eye(L.rows, R.rows, CV_32F) * (0.01 / norm(R));
  float alpha = 0.03;
  cv::Mat  res;
  cout << "Begin solving .....";
  int numsample = R.cols;
  for (int rr = 1; rr < 100; rr++)
  {
    int nbatch = 6;
    for (int i = 0; i < numsample; i+= nbatch)
    {
      int iend = MIN(i + nbatch, numsample);
      Mat rcol = R.colRange(i, iend);
      Mat lcol = L.colRange(i, iend);
 
      res = lcol - M*rcol;
      M = M + (alpha / rr / nbatch) *res*rcol.t();
    }
    res = L - M*R;
    cout << "|" << cv::norm(res, cv::NORM_L2);
  }
  cout << " ....done." << endl;
  return M;
}
inline cv::Mat shape2d_to_mat(vector<Point2f>&pts)
{
  int numpts = pts.size();
  Mat v(numpts * 2, 1, CV_32F);
  for (int j = 0; j < numpts; j++)
  {
    v.at<float>(2 * j) = pts[j].x;
    v.at<float>(2 * j + 1) = pts[j].y;
  }
  return v;
}
inline shape2d mat_to_shape2d(Mat & v)
{
  int numpts = v.rows / 2;
  assert(v.cols == 1);
  shape2d pts(numpts, cv::Point2f(0, 0));

  for (int j = 0; j < numpts; j++)
  {
    pts[j].x = v.at<float>(2 * j);
    pts[j].y = v.at<float>(2 * j + 1);
  }
  return pts;
}
vector<cv::Point2f> read_pts_landmarks(std::string filename)
{
  using std::getline;
  vector<cv::Point2f> landmarks;
  landmarks.reserve(68);

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(string("Could not open landmark file: " + filename));
  }

  string line;
  // Skip the first 3 lines, they're header lines:
  getline(file, line); // 'version: 1'
  getline(file, line); // 'n_points : 68'
  getline(file, line); // '{'

  while (getline(file, line))
  {
    if (line == "}") { // end of the file
      break;
    }
    std::stringstream line_stream(line);
    cv::Point2f landmark(0.0f, 0.0f);
    if (!(line_stream >> landmark.x >> landmark.y)) {
      throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
    }
    landmark.x -= 1.0f;
    landmark.y -= 1.0f;
    landmarks.emplace_back(landmark);
  }
  return landmarks;
};
void read_names_pair(const string &strListName, const string &strFilePath,const string& pts_ext, const string& img_ext,
  vector<string> &vstrShapeName, vector<string> &vstrImageName)
{
  ifstream fNameFile;
  fNameFile.open(strListName.c_str());
  vstrShapeName.reserve(2000);
  vstrImageName.reserve(2000);
  while (!fNameFile.eof())
  {
    string s;
    getline(fNameFile, s);
    if (!s.empty())
    {
      stringstream ss;
      ss << s;
      string ptname = strFilePath + "/" + ss.str() + pts_ext;
      string imagename = ptname.substr(0, ptname.find(".pts")) + img_ext;

      // cout << ptname << imagename<<endl;
      vstrShapeName.push_back(ptname);
      vstrImageName.push_back(imagename);
    }
  }
}
void train_map_facerect_and_eyes(vector<string>& vstrPts, vector<string>& vstrImg)
{ 
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
  cv::Mat M ; //L=M*R
  for (int i = 0; i < numsamples/*vstrPts.size()*/; i++)
  {
    vector<cv::Point2f> pts = read_pts_landmarks(vstrPts[i]);
    cv::Mat img = cv::imread(vstrImg[i], 0);

    float scale = 1.0;
    if (img.rows <= img.cols && img.rows > 200) scale = 120.0 / img.rows;
    if (img.rows > img.cols && img.cols > 200) scale = 160.0 / img.cols;

    cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_LINEAR);
    vector<cv::Rect> detected_faces;
    face_cascade.detectMultiScale(img, detected_faces, 1.2, 4, 0, cv::Size(50, 50));

    int num = detected_faces.size();
    if (num == 0)
      continue;

    //cv::rectangle(img, detected_faces[0], cv::Scalar(0, 0, 255), 1);
    //for (int j = 0; j < pts.size(); j++)    {  pts[j] = pts[j]*scale;  } 
    //cv::rectangle(img, cv::boundingRect(pts), cv::Scalar(255, 255, 255), 1);
    //for (int j = 0; j < pts.size(); j++)    {  cv::circle(img, cv::Point(pts[j].x, pts[j].y), 2, cv::Scalar(0, 0, 255), -1);    }

    cv::Point2f p0 = (pts[eyes_index[0]] + pts[eyes_index[1]])*scale / 2;
    cv::Point2f p1 = (pts[eyes_index[2]] + pts[eyes_index[3]])*scale / 2;

    cv::Rect rectface = detected_faces[0];
    if (rectface.contains(p0) && rectface.contains(p1))
    {
      good++;
      // cv::circle(img, cv::Point(p0.x, p0.y), 2, cv::Scalar(255, 0, 255), -1);
      // cv::circle(img, cv::Point(p1.x, p1.y), 2, cv::Scalar(255, 0, 255), -1);
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
 // cout << "M_rect_to_eyes:\n" << M_eyes_to_rect.inv() << endl << endl;
}
void preprocess_images(vector<string>& vstrPts, vector<string>& vstrImg)
{ 
  cv::CascadeClassifier face_cascade;
  face_cascade.load(face_model);
 
  bool showimg = false; 
  Mat imcanvas;

  cout << "Collecting cropped "<< vstrPts.size() <<" faces and normlized shapes ..." << endl;

  int count = 0;
  for (int i = 0; i < vstrPts.size(); i++)
  {
    vector<cv::Point2f> pts = read_pts_landmarks(vstrPts[i]);
    cv::Mat img_ = cv::imread(vstrImg[i], 0);//-1 origimage,0 grayscale
    cv::Mat img;
    float scale = 1.0;
    if (img_.rows <= img_.cols && img_.rows > 400) scale = 240.0 / img_.rows;
    if (img_.rows > img_.cols && img_.cols > 400) scale = 320.0 / img_.cols;
    cv::resize(img_, img, cv::Size(), scale, scale, cv::INTER_LINEAR);

    vector<cv::Rect> detected_faces;
    face_cascade.detectMultiScale(img, detected_faces, 1.2, 4, 0, cv::Size(50, 50));

    cv::Point2f p0 = (pts[eyes_index[0]] + pts[eyes_index[1]])*scale / 2;
    cv::Point2f p1 = (pts[eyes_index[2]] + pts[eyes_index[3]])*scale / 2;

    int idx = -1;
    for (int i = 0; i < detected_faces.size(); i++)
    {
      //eye lmks in this face rectangle
      if (detected_faces[i].contains(p0) && detected_faces[i].contains(p1))
        idx = i;
    }
    //cout << "in image :" << i << ", idx of good face:" << idx << endl;

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

    float scale_again = (float)normal_face_width / (float)rectface.width;
    rectface = cv::Rect(0.5 + (float)rectface.x*scale_again, 0.5 + (float)rectface.y*scale_again, normal_face_width, normal_face_width);
    scale *= scale_again;
    cv::resize(img_, img, cv::Size(), scale, scale, cv::INTER_LINEAR);

    //////crop face region   
    int normal_roi_width = normal_face_width + border * 2;
    cv::Mat roiImg ;
    int left = rectface.x, right = left + rectface.width - 1;
    int top = rectface.y, low = top + rectface.height - 1;

    int bdleft = MAX(border - left, 0);
    int bdtop = MAX(border - top, 0);
    int bdright = MAX(right + border - img.cols + 1, 0);
    int bdlow = MAX(low + border - img.rows + 1, 0);

    if (bdleft > 0 || bdtop > 0 || bdright > 0 || bdlow > 0)
    {
      cv::Mat extendedImage;
      cv::copyMakeBorder(img, extendedImage, bdtop, bdlow, bdleft, bdright, cv::BORDER_CONSTANT, cv::Scalar(0));
      cv::Rect roi((left - border) + bdleft, (top - border) + bdtop, normal_roi_width, normal_roi_width);
      roiImg = extendedImage(roi).clone();
    }
    else
    {
      cv::Rect roi(left - border, top - border, normal_roi_width, normal_roi_width);
      roiImg = img(roi).clone();
    }

    //renew lmks
    for (int j = 0; j < pts.size(); j++)
    {
      pts[j].x = pts[j].x * scale - (left - border);
      pts[j].y = pts[j].y * scale - (top - border);
    }

   //here is a chance to shorten the shape vector 
    shape2d pt_few;
    pt_few.resize(vshortlmk.size());
    for (int j = 0; j <vshortlmk.size(); j++)
      pt_few[j] = pts[vshortlmk[j]];
    pts = pt_few; 

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
      HOGDescriptor hog = hogs[0];
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
      sprintf(name, "./crop/f_%d.png", i);
      imwrite(name, imcanvas);
      imshow("Face ROI", imcanvas);
      waitKey(10);
    }
  }//end collecting

  //do averaging 
  for (int j = 0; j < shape_mean.size(); j++)
    shape_mean[j] = shape_mean[j] / count;

  if (showimg)
  {
    cv::Mat avgimg,temp;
    train_samples[0].convertTo(avgimg, CV_32FC1);

    for (int i = 1; i < train_samples.size(); i++)
    {
      train_samples[i].convertTo(temp, CV_32FC1);
      avgimg = avgimg + temp;
    }
    avgimg = avgimg / train_samples.size();  
   
    HOGDescriptor hog = hogs[0];
    int half_wid = hog.blockSize.width >> 1;
    avgimg.convertTo(imcanvas, CV_8UC1);
    cvtColor(imcanvas, imcanvas, COLOR_GRAY2RGB);

    for (int j = 0; j < shape_mean.size(); j++)
    {
      cv::circle(imcanvas, cv::Point(shape_mean[j].x, shape_mean[j].y), 2, cv::Scalar(0,0,255), -1);
      Rect r(shape_mean[j].x - half_wid, shape_mean[j].y - half_wid, hog.blockSize.width, hog.blockSize.width);
      cv::rectangle(imcanvas, r, cv::Scalar(0, 255, 0), 1);
    }
    cout << "Press any to continue." << endl;
    imshow("meanface with shape_mean", imcanvas);
    waitKey();
  }
}
inline vector<float> extract_HOG_descriptor(Mat & roiImg, vector<Point2f>& pt_shape, HOGDescriptor& hog )
{
  int half_wid = hog.blockSize.width >> 1;
  vector<float> des;
  vector<Point> pos; //need INT pos values
  for (int j = 0; j < pt_shape.size(); j++)
    pos.push_back(Point(pt_shape[j].x - half_wid + 0.5, pt_shape[j].y - half_wid + 0.5));

  Size winStride = hog.blockSize;
  hog.compute(roiImg, des, winStride, Size(0, 0), pos);
  des.push_back(1);
  return des;
}
bool mats_write(const char* filename, vector<Mat>& vM)
{
  FILE* file = fopen(filename, "wb");
  if (file == NULL || vM.empty())
    return false;

  int num = vM.size();
  fwrite(&num, sizeof(int), 1, file);

  for (int i = 0; i<num; i++)
  {
    Mat M = vM[i];// cout << M.step << " " << M.cols << "*" << M.rows << endl;
    int headData[3] = { M.cols, M.rows, M.type() };
    fwrite(headData, sizeof(int), 3, file);
    fwrite(M.data, sizeof(char), M.step * M.rows, file);
  }
  fclose(file);
  return true;
}
vector<Mat> mats_read(const char* filename)
{
  vector<Mat> vM;
  FILE* file = fopen(filename, "rb");
  if (file == NULL)
    return vM;

  int num;
  fread(&num, sizeof(int), 1, file);

  for (int i = 0; i<num; i++)
  {
    int headData[3];
    fread(headData, sizeof(int), 3, file);
    Mat M(headData[1], headData[0], headData[2]);
    fread(M.data, sizeof(char), M.step * M.rows, file);
    vM.push_back(M);
  }
  fclose(file);
  return vM;
}
void train_by_regress()
{
  int numsample_test = 100; //left some for test
  int numsample = train_shapes.size()- numsample_test;
  int numpts = shape_mean.size();

  cout << "Training SDM on "<< numsample <<" images, left " << numsample_test << " for testing ..." << endl;
  
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
      vector<float> des = extract_HOG_descriptor(roiImg, shape_current_all[i], hogs[r]);
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
  
  mats_write("face_sdm.bin", Mreg);
  cout << "Write to sdm.bin" << endl;

  if (1)
  {
    cout << "Testing on unseen images ..........\n";
    Mat imcanvas;
    for (int i = numsample; i < train_samples.size(); i++)//on rest image in trainset
    {
      Mat roiImg = train_samples[i];
      cvtColor(roiImg, imcanvas, COLOR_GRAY2RGB);
            
      for (int j = 0; j < shape_mean.size(); j++)
        cv::circle(imcanvas, cv::Point(shape_mean[j].x, shape_mean[j].y), 2, cv::Scalar(255,0,0), -1);
     
      shape2d pts = shape_mean;
      for (int r = 0; r < Mreg.size(); r++)
      {
        vector<float> des = extract_HOG_descriptor(roiImg, pts, hogs[r]);
       
        Mat rr = Mat(des.size(), 1, CV_32F, des.data());
        Mat v_shape = Mreg[r] *rr + shape2d_to_mat(pts);
        pts = mat_to_shape2d(v_shape);

        if(r==0)
        for (int j = 0; j < pts.size(); j++) 
          cv::circle(imcanvas, cv::Point(pts[j].x, pts[j].y), 2, cv::Scalar(0, 0, 255), -1); 
      }
      for (int j = 0; j < pts.size(); j++)
        cv::circle(imcanvas, cv::Point(pts[j].x, pts[j].y), 2, cv::Scalar(0, 255, 0), -1);

      char name[200];
      sprintf(name, "./crop/test_%d.png", i);
      imwrite(name, imcanvas);
     // imshow("Test on unseen", imcanvas);
    //  waitKey();
    }
  } 
}

void main()
{
  set_HOGs();
  string imagepath = "./helen/trainset";
  string ptlistname = "./helen/trainset/ptlist.txt"; //dir *.pts/b > ptlist.txt
  vector<string> vstrPts, vstrImg;

  read_names_pair(ptlistname, imagepath, string(), ".jpg", vstrPts, vstrImg);
  train_map_facerect_and_eyes(vstrPts, vstrImg);

  preprocess_images(vstrPts, vstrImg);
  train_by_regress();
  //TODO : load sdm model to test on /testset
  vector<cv::Mat> Mreg = mats_read("face_sdm.bin");
}