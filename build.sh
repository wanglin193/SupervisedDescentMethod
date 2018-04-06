cd bin/data/helen

cd trainset
ls *.pts -1>ptlist.txt
cd ..

cd testset
ls *.pts -1>ptlist.txt
cd ..

cd ../../..


mkdir build
cd build
cmake ..
make 


