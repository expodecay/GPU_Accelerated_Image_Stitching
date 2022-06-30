//
// Created by sean on 6/28/2022.
//

#ifndef IMAGE_STITCHING_MYBLEND_H
#define IMAGE_STITCHING_MYBLEND_H

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

class MyBlend : public cv::detail::Blender{
public:
    CV_WRAP MyBlend(int try_gpu = false, int num_bands = 5, int weight_type = CV_32F);

    CV_WRAP int numBands() const { return actual_num_bands_; }
    CV_WRAP void setNumBands(int val) { actual_num_bands_ = val; }

    CV_WRAP void prepare(cv::Rect dst_roi) CV_OVERRIDE;
    CV_WRAP void feed(cv::InputArray img, cv::InputArray mask, cv::Point tl) CV_OVERRIDE;
    CV_WRAP void blend(CV_IN_OUT cv::InputOutputArray dst, CV_IN_OUT cv::InputOutputArray dst_mask) CV_OVERRIDE;

private:
    int actual_num_bands_, num_bands_;
    std::vector<cv::UMat> dst_pyr_laplace_;
    std::vector<cv::UMat> dst_band_weights_;
    cv::Rect dst_roi_final_;
    bool can_use_gpu_;
    int weight_type_; //CV_32F or CV_16S
#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    std::vector<cv::cuda::GpuMat> gpu_dst_pyr_laplace_;
    std::vector<cv::cuda::GpuMat> gpu_dst_band_weights_;
    std::vector<cv::Point> gpu_tl_points_;
    std::vector<cv::cuda::GpuMat> gpu_imgs_with_border_;
    std::vector<std::vector<cv::cuda::GpuMat> > gpu_weight_pyr_gauss_vec_;
    std::vector<std::vector<cv::cuda::GpuMat> > gpu_src_pyr_laplace_vec_;
    std::vector<std::vector<cv::cuda::GpuMat> > gpu_ups_;
    cv::cuda::GpuMat gpu_dst_mask_;
    cv::cuda::GpuMat gpu_mask_;
    cv::cuda::GpuMat gpu_img_;
    cv::cuda::GpuMat gpu_weight_map_;
    cv::cuda::GpuMat gpu_add_mask_;
    int gpu_feed_idx_;
    bool gpu_initialized_;
#endif
};


#endif //IMAGE_STITCHING_MYBLEND_H
