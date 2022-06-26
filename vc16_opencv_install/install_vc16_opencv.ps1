$prefix_var='' # directory prefix for opencv
$opencv_contrib_location_var='' # opencv contribute module location
$opencv_version=''

#

$use_opencv_version = Read-Host -Prompt 'Use specific opencv version? (y/n) - defaults to no and uses the latest: '

if ( $use_opencv_version -ieq 'y' )
{
	$opencv_version = Read-Host -Prompt "which opencv version?: "
}

# check to see if the user wants to set a prefix and/or set a different location for the opencv_contrib files

$use_prefix = Read-Host -Prompt 'would you like to use a directory prefix (y/n) - defaults to no? ' 

if ( $use_prefix -ieq 'y' )
{
	$prefix_var = Read-Host -Prompt 'install_prefix (example: C:\Users\sean\opencv\local_opencv): ' 
}

$use_custom_contrib_location = Read-Host -Prompt 'would you like to set opencv_contrib location (y/n) - defaults to no? ' 

if ( $use_custom_contrib_location -ieq 'y' )
{
	$opencv_contrib_location_var = Read-Host -Prompt 'set full path location of opencv_contrib (example: C:\Users\sean\opencv\opencv_contrib\modules): ' 
}

if ( $opencv_version -ieq "" )
{
	git clone https://github.com/opencv/opencv.git
}
else
{
	git clone https://github.com/opencv/opencv.git -b $opencv_version
}

cd opencv

$opencv_dir=$(Get-Location) # get opencv root directory path
$opencv_dir=$opencv_dir.path

if ( $opencv_version -ieq "" )
{
	git clone https://github.com/opencv/opencv_contrib.git
}
else
{
	git clone https://github.com/opencv/opencv_contrib.git -b $opencv_version
}

# create build folder and depending on the options have cmake generate the build files

mkdir build ; cd build

if ( $opencv_contrib_location_var -ieq "" )
{
	echo "use default opencv_contrib path!"
	$opencv_contrib_location_var="$opencv_dir\opencv_contrib\modules"
}

if ( $prefix_var -ine "" )
{
	cmake -G "Visual Studio 16 2019" -A x64 -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=OFF -D BUILD_opencv_python_bindings_generator=ON -D CUDA_ARCH_BIN="8.6" -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$prefix_var -D WITH_CUDA=ON -D CUDA_GENERATION=Auto -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D BUILD_TIFF=ON -D ENABLE_CXX11=ON -D INSTALL_C_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH="$opencv_contrib_location_var" -D OPENCV_ENABLE_NONFREE=ON -D BUILD_EXAMPLES=ON -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..
}
else
{
	cmake -G "Visual Studio 16 2019" -A x64 -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=OFF -D BUILD_opencv_python_bindings_generator=ON -D CUDA_ARCH_BIN="8.6" -D CMAKE_BUILD_TYPE=Release -D WITH_CUDA=ON -D CUDA_GENERATION=Auto -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D BUILD_TIFF=ON -D ENABLE_CXX11=ON -D INSTALL_C_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH="$opencv_contrib_location_var" -D OPENCV_ENABLE_NONFREE=ON -D BUILD_EXAMPLES=ON -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..
}

# build opencv

cmake --build . --target ALL_BUILD --config Release

# install opencv

cmake --build . --target INSTALL --config Release

echo ""
echo "open visual studio project found in opencv/build/x64. Right-Click on INSTALL which is under the CMakeTargets folder (in the solution explorer) and select build from the list."
echo "this will install the dll and lib files needed to compile a new project with opencv. This will also copy the opencv python functions to your active python interpreter so you can"
echo "functions there."

echo "Simple example in python to make sure this works..."
echo "import cv2"
echo "cv2.cuda.getCudaEnabledDeviceCount()"
echo ""
echo "The above python should return a 1 if you have cuda installed and also have a nvidia gpu"
