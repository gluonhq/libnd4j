#!bin/sh
mkdir -p target/ios-x86_64
mkdir -p target/ios-arm64
./buildnativeoperations.sh -o ios-x86_64 
cp blasbuild/cpu/blas/libnd4jcpu.a target/ios-x86_64
./buildnativeoperations.sh -o ios-arm64 
cp blasbuild/cpu/blas/libnd4jcpu.a target/ios-arm64
lipo -create -output target/libnd4jcpu.a target/ios-x86_64/libnd4jcpu.a target/ios-arm64/libnd4jcpu.a
