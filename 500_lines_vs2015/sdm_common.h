using namespace std;
using namespace cv;

#define HOG_GLOBAL -1
#define HOG_ALL 0

typedef vector<cv::Point2f> shape2d;

void set_global_hog(HOGDescriptor&hog, int roi_size, int pix_in_cell, int cell_in_block, int n_orient)
{
  int cell_in_win = (int)((float)roi_size / (float)pix_in_cell);
  hog.winSize = Size(cell_in_win*pix_in_cell, cell_in_win*pix_in_cell);
  hog.cellSize = Size(pix_in_cell * cell_in_block, pix_in_cell*cell_in_block);
  hog.blockSize = cv::Size(cell_in_block * hog.cellSize.width, cell_in_block * hog.cellSize.height);
  hog.blockStride = hog.cellSize;
  hog.nbins = n_orient;
}
void set_hog_params(HOGDescriptor&hog, int pix_in_cell, int cell_in_block, int n_orient)
{
  hog.cellSize = Size(pix_in_cell, pix_in_cell);
  hog.blockSize = cv::Size(cell_in_block * hog.cellSize.width, cell_in_block * hog.cellSize.height);
  hog.winSize = hog.blockSize; //this define feature region
  hog.blockStride = hog.winSize;//useless when set hog.winSize = hog.blockSize; 
  hog.nbins = n_orient;
}
void set_HOGs(vector<HOGDescriptor>& hogs,int global_size,vector<int>& vnum_lmks)
{
  int num_hog = hogs.size();
  int pix_in_cell = (int)(global_size*0.1 + 0.5);

  if (vnum_lmks[0] == HOG_GLOBAL)
    set_global_hog(hogs[0], global_size, 8, 4, 9);
  else
    set_hog_params(hogs[0], pix_in_cell, 4, 6);

  if (num_hog > 1) set_hog_params(hogs[1], pix_in_cell, 4, 6);
  if (num_hog > 2) set_hog_params(hogs[2], pix_in_cell, 4, 4);
  for (int i = 3; i<num_hog; i++)
    set_hog_params(hogs[i], pix_in_cell, 3, 4);
}

vector<Mat> mats_read(const char* filename)
{
  vector<Mat> vM;
  FILE* file = fopen(filename, "rb");
  if (file == NULL)
    return vM;

  int num;
  fread(&num, sizeof(int), 1, file);

  for (int i = 0; i < num; i++)
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
bool mats_write(const char* filename, vector<Mat>& vM)
{
  FILE* file = fopen(filename, "wb");
  if (file == NULL || vM.empty())
    return false;

  int num = vM.size();
  fwrite(&num, sizeof(int), 1, file);

  for (int i = 0; i < num; i++)
  {
    Mat M = vM[i];// cout << M.step << " " << M.cols << "*" << M.rows << endl;
    int headData[3] = { M.cols, M.rows, M.type() };
    fwrite(headData, sizeof(int), 3, file);
    fwrite(M.data, sizeof(char), M.step * M.rows, file);
  }
  fclose(file);
  return true;
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

shape2d read_pts_landmarks(const std::string filename)
{
  using std::getline;
  shape2d landmarks;
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
void read_names_pair(const string &strListName, const string &strFilePath,
  const string& pts_ext, const string& img_ext,
  vector<string> &vstrShapeName, vector<string> &vstrImageName)
{
  ifstream fNameFile;
  fNameFile.open(strListName.c_str());
  vstrShapeName.clear();
  vstrImageName.clear();
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

      vstrShapeName.push_back(ptname);
      vstrImageName.push_back(imagename);
    }
  }
}

vector<float> extract_HOG_descriptor(Mat & roiImg, vector<Point2f>& pt_shape, HOGDescriptor& hog, int num_idx_lmk, vector<int>& idx_lmk)
{
  int half_wid = hog.winSize.width >> 1;
  vector<float> des;
  vector<Point> pos;
  if (num_idx_lmk == HOG_GLOBAL) //face region
  {
    pos.push_back(Point(roiImg.cols / 2 - half_wid, roiImg.rows / 2 - half_wid));
  }
  else if (num_idx_lmk == HOG_ALL)
  {
    for (int j = 0; j < pt_shape.size(); j++)
      pos.push_back(Point(pt_shape[j].x - half_wid + 0.5, pt_shape[j].y - half_wid + 0.5));
  }
  else
  {
    for (int j = 0; j < num_idx_lmk; j++)
    {
      Point2f pcorner = pt_shape[idx_lmk[j]] - Point2f(half_wid, half_wid);
      pos.push_back(Point(pcorner.x + 0.5, pcorner.y + 0.5));
    }
  }
  hog.compute(roiImg, des, Size(0, 0), Size(0, 0), pos);  //Size winStride = Size(0, 0);
  des.push_back(1.0);
  return des;
}

void draw_shape(Mat& imcanvas, shape2d& pts, cv::Scalar color)
{
  for (int j = 0; j < pts.size(); j++)
    cv::circle(imcanvas, cv::Point(pts[j].x, pts[j].y), 2, color, -1);
}
