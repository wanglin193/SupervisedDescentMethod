cd bin/data/helen

cd trainset
ls *.pts -1>ptlist.txt
cd ..

cd testset
ls *.pts -1>ptlist.txt
cd ../../../..

cd bin
mkdir output
cd ..

mkdir build
cd build
cmake ..
make 


