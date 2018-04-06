# SDM for face alignment

1. Try training result on test set
~~~
cd ./bin
./test_sdm_facealign
~~~

2. Create pts file name list for load training images, use 
~~~
cd trainset
ls *.pts -1>ptlist.txt
~~~
   or in Windows use
~~~
cd trainset
dir *.pts/b>ptlist.txt
~~~

3. Some results: 

  * blue dots: init position (meanshape), same for all 

  * red dos: after first round descent iteration 

  * green dots: after all five rounds descent iteration 

![test_tile](https://github.com/wanglin193/SupervisedDescentMethod/blob/master/crop/test_tile.jpg)



