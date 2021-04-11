echo "Uncompress ORB vocabulary ..."

cd vocabularies
tar -xf orb.txt.tar.gz
cd ..

echo "Running ORB_SLAM2 build script ..."

cd thirdparty/ORB_SLAM2/
chmod +x build.sh
./build.sh

echo "Running SuperPoint_SLAM build script ..."

cd ../SuperPoint_SLAM/
chmod +x build.sh
./build.sh