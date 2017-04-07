# Train an SDM for face alignment in 500 lines of C++ code

* The theory is based on the idea of _Supervised Descent Method and Its Applications to Face Alignment_, from X. Xiong & F. De la Torre, CVPR 2013

* OpenCV's Haar-based face detection only has around 75% detection rate, so for those in trainset fail to detect faces, the face rectangle shoud be initialized by annotated eyes' postions. So I leanrn a matrix to map eyes' postions to face rectangle.

* OpenCV ```cv::HOGDescriptor``` is used to extract HOG feature descriptors around face landmarks' region like this:

``` c++
Mat Img;
vector<Point2f> pt_shape;
HOGDescriptor hog;
int half_wid = hog.blockSize.width/2;
vector<float> des;

vector<Point> pos;
for (int j = 0; j < pt_shape.size(); j++)
    pos.push_back(Point(pt_shape[j].x - half_wid + 0.5, pt_shape[j].y - half_wid + 0.5));

hog.compute(Img, des, winStride, Size(0, 0), pos);
```

* Some samples in trainset are used for testing the model, so the preprocessing steps are same for all images: crop face region (by face detection or by eyes' position) and align it to same size, then add borders to extract HOGs.

* Some results (only 22 landmarks used here): 

  * blue dots: init position (meanshape), same for all 

  * red dos: after first round descent iteration 

  * green dots: after all five rounds descent iteration 

![test_tile](https://github.com/wanglin193/SupervisedDescentMethod/blob/master/crop/test_tile.png)



