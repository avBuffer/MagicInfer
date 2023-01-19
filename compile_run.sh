
echo "***** building source *****"
rm -rf build log out
mkdir build log

cd build
cmake .. && make -j16
echo ""


echo "***** runing test *****"

cd test
./test_magic &> ../../log/test.log 
cd ..


echo "***** runing bench *****"
cd bench
./bench_magic &> ../../log/bench.log
cd ..


echo "***** runing demo *****"
cd demo
./yolo_infer &> ../../log/yolo.log
cd ../../


echo "***** end test/bench/demo *****"
echo ""
