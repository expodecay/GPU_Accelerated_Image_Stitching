//
// Created by sean on 6/28/2022.
//

#include "MyBlend.h"

#define USE_CUDA_FOR_PROJECT true

#if USE_CUDA_FOR_PROJECT
#define HAVE_CUDA
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/stitching/warpers.hpp>
#endif

namespace cv { namespace cuda { namespace device { namespace blend {

                void addSrcWeightGpu16S(const PtrStep<short> src, const PtrStep<short> src_weight,
                                        PtrStep<short> dst, PtrStep<short> dst_weight, Rect &rc);

                void addSrcWeightGpu32F(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
                                        cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, cv::Rect &rc);

                void normalizeUsingWeightMapGpu16S(const cv::cuda::PtrStep<short> weight, cv::cuda::PtrStep<short> src,
                                                   const int width, const int height);

                void normalizeUsingWeightMapGpu32F(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<short> src,
                                                   const int width, const int height);
}}}}

namespace {
    static const float WEIGHT_EPS = 1e-5f;
}

MyBlend::MyBlend(int try_gpu, int num_bands, int weight_type)
{
    num_bands_ = 0;
    setNumBands(num_bands);

#if defined(HAVE_CUDA) && defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    can_use_gpu_ = try_gpu && cv::cuda::getCudaEnabledDeviceCount();
    gpu_feed_idx_ = 0;
#else
    CV_UNUSED(try_gpu);
    can_use_gpu_ = false;
#endif

    CV_Assert(weight_type == CV_32F || weight_type == CV_16S);
    weight_type_ = weight_type;
}

void MyBlend::prepare(cv::Rect dst_roi)
{
    dst_roi_final_ = dst_roi;

    // Crop unnecessary bands
    double max_len = static_cast<double>(std::max(dst_roi.width, dst_roi.height));
    num_bands_ = std::min(actual_num_bands_, static_cast<int>(ceil(std::log(max_len) / std::log(2.0))));

    // Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
    dst_roi.width += ((1 << num_bands_) - dst_roi.width % (1 << num_bands_)) % (1 << num_bands_);
    dst_roi.height += ((1 << num_bands_) - dst_roi.height % (1 << num_bands_)) % (1 << num_bands_);

    Blender::prepare(dst_roi);

#if defined(HAVE_CUDA) && defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    if (can_use_gpu_)
    {
        gpu_initialized_ = false;
        gpu_feed_idx_ = 0;

        gpu_tl_points_.clear();
        gpu_weight_pyr_gauss_vec_.clear();
        gpu_src_pyr_laplace_vec_.clear();
        gpu_ups_.clear();
        gpu_imgs_with_border_.clear();

        gpu_dst_pyr_laplace_.resize(num_bands_ + 1);
        gpu_dst_pyr_laplace_[0].create(dst_roi.size(), CV_16SC3);
        gpu_dst_pyr_laplace_[0].setTo(cv::Scalar::all(0));

        gpu_dst_band_weights_.resize(num_bands_ + 1);
        gpu_dst_band_weights_[0].create(dst_roi.size(), weight_type_);
        gpu_dst_band_weights_[0].setTo(0);

        for (int i = 1; i <= num_bands_; ++i)
        {
            gpu_dst_pyr_laplace_[i].create((gpu_dst_pyr_laplace_[i - 1].rows + 1) / 2,
                (gpu_dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
            gpu_dst_band_weights_[i].create((gpu_dst_band_weights_[i - 1].rows + 1) / 2,
                (gpu_dst_band_weights_[i - 1].cols + 1) / 2, weight_type_);
            gpu_dst_pyr_laplace_[i].setTo(cv::Scalar::all(0));
            gpu_dst_band_weights_[i].setTo(0);
        }
    }
    else
#endif
    {
        dst_pyr_laplace_.resize(num_bands_ + 1);
        dst_pyr_laplace_[0] = dst_;

        dst_band_weights_.resize(num_bands_ + 1);
        dst_band_weights_[0].create(dst_roi.size(), weight_type_);
        dst_band_weights_[0].setTo(0);

        for (int i = 1; i <= num_bands_; ++i)
        {
            dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2,
                                       (dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
            dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2,
                                        (dst_band_weights_[i - 1].cols + 1) / 2, weight_type_);
            dst_pyr_laplace_[i].setTo(cv::Scalar::all(0));
            dst_band_weights_[i].setTo(0);
        }
    }
}

void MyBlend::feed(cv::InputArray _img, cv::InputArray mask, cv::Point tl)
{
#if ENABLE_LOG
    int64_t t = cv::getTickCount();
#endif

    cv::UMat img;

#if defined(HAVE_CUDA) && defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    // If using gpu save the top left coordinate when running first time after prepare
    if (can_use_gpu_)
    {
        if (!gpu_initialized_)
        {
            gpu_tl_points_.push_back(tl);
        }
        else
        {
            tl = gpu_tl_points_[gpu_feed_idx_];
        }
    }
    // If _img is not a GpuMat get it as UMat from the InputArray object.
    // If it is GpuMat make a dummy object with right dimensions but no data and
    // get _img as a GpuMat
    if (!_img.isGpuMat())
#endif
    {
        img = _img.getUMat();
    }
#if defined(HAVE_CUDA) && defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    else
    {
        gpu_img_ = _img.getGpuMat();
        img = cv::UMat(gpu_img_.rows, gpu_img_.cols, gpu_img_.type());
    }
#endif

    CV_Assert(img.type() == CV_16SC3 || img.type() == CV_8UC3);
    CV_Assert(mask.type() == CV_8U);

    // Keep source image in memory with small border
    int gap = 3 * (1 << num_bands_);
    cv::Point tl_new(std::max(dst_roi_.x, tl.x - gap),
                 std::max(dst_roi_.y, tl.y - gap));
    cv::Point br_new(std::min(dst_roi_.br().x, tl.x + img.cols + gap),
                 std::min(dst_roi_.br().y, tl.y + img.rows + gap));

    // Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
    // After that scale between layers is exactly 2.
    //
    // We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
    // image is bordered to have size equal to the final image size, but this is too memory hungry approach.
    tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> num_bands_) << num_bands_);
    tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> num_bands_) << num_bands_);
    int width = br_new.x - tl_new.x;
    int height = br_new.y - tl_new.y;
    width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_);
    height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_);
    br_new.x = tl_new.x + width;
    br_new.y = tl_new.y + height;
    int dy = std::max(br_new.y - dst_roi_.br().y, 0);
    int dx = std::max(br_new.x - dst_roi_.br().x, 0);
    tl_new.x -= dx; br_new.x -= dx;
    tl_new.y -= dy; br_new.y -= dy;

    int top = tl.y - tl_new.y;
    int left = tl.x - tl_new.x;
    int bottom = br_new.y - tl.y - img.rows;
    int right = br_new.x - tl.x - img.cols;

#if defined(HAVE_CUDA) && defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    if (can_use_gpu_)
    {
        if (!gpu_initialized_)
        {
            gpu_imgs_with_border_.push_back(cv::cuda::GpuMat());
            gpu_weight_pyr_gauss_vec_.push_back(std::vector<cv::cuda::GpuMat>(num_bands_+1));
            gpu_src_pyr_laplace_vec_.push_back(std::vector<cv::cuda::GpuMat>(num_bands_+1));
            gpu_ups_.push_back(std::vector<cv::cuda::GpuMat>(num_bands_));
        }

        // If _img is not GpuMat upload it to gpu else gpu_img_ was set already
        if (!_img.isGpuMat())
        {
            gpu_img_.upload(img);
        }

        // Create the source image Laplacian pyramid
        cv::cuda::copyMakeBorder(gpu_img_, gpu_imgs_with_border_[gpu_feed_idx_], top, bottom,
                             left, right, cv::BORDER_REFLECT);
        gpu_imgs_with_border_[gpu_feed_idx_].convertTo(gpu_src_pyr_laplace_vec_[gpu_feed_idx_][0], CV_16S);
        for (int i = 0; i < num_bands_; ++i)
            cv::cuda::pyrDown(gpu_src_pyr_laplace_vec_[gpu_feed_idx_][i],
                          gpu_src_pyr_laplace_vec_[gpu_feed_idx_][i + 1]);
        for (int i = 0; i < num_bands_; ++i)
        {
            cv::cuda::pyrUp(gpu_src_pyr_laplace_vec_[gpu_feed_idx_][i + 1], gpu_ups_[gpu_feed_idx_][i]);
            cv::cuda::subtract(gpu_src_pyr_laplace_vec_[gpu_feed_idx_][i],
                           gpu_ups_[gpu_feed_idx_][i],
                           gpu_src_pyr_laplace_vec_[gpu_feed_idx_][i]);
        }

        // Create the weight map Gaussian pyramid only if not yet initialized
        if (!gpu_initialized_)
        {
            if (mask.isGpuMat())
            {
                gpu_mask_ = mask.getGpuMat();
            }
            else
            {
                gpu_mask_.upload(mask);
            }

            if (weight_type_ == CV_32F)
            {
                gpu_mask_.convertTo(gpu_weight_map_, CV_32F, 1. / 255.);
            }
            else // weight_type_ == CV_16S
            {
                gpu_mask_.convertTo(gpu_weight_map_, CV_16S);
                cv::cuda::compare(gpu_mask_, 0, gpu_add_mask_, cv::CMP_NE);
                cv::cuda::add(gpu_weight_map_, cv::Scalar::all(1), gpu_weight_map_, gpu_add_mask_);
            }
            cv::cuda::copyMakeBorder(gpu_weight_map_, gpu_weight_pyr_gauss_vec_[gpu_feed_idx_][0], top,
                                 bottom, left, right, cv::BORDER_CONSTANT);
            for (int i = 0; i < num_bands_; ++i)
                cv::cuda::pyrDown(gpu_weight_pyr_gauss_vec_[gpu_feed_idx_][i],
                              gpu_weight_pyr_gauss_vec_[gpu_feed_idx_][i + 1]);
        }

        int y_tl = tl_new.y - dst_roi_.y;
        int y_br = br_new.y - dst_roi_.y;
        int x_tl = tl_new.x - dst_roi_.x;
        int x_br = br_new.x - dst_roi_.x;

        // Add weighted layer of the source image to the final Laplacian pyramid layer
        for (int i = 0; i <= num_bands_; ++i)
        {
            cv::Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl);
            cv::cuda::GpuMat &_src_pyr_laplace = gpu_src_pyr_laplace_vec_[gpu_feed_idx_][i];
            cv::cuda::GpuMat _dst_pyr_laplace = gpu_dst_pyr_laplace_[i](rc);
            cv::cuda::GpuMat &_weight_pyr_gauss = gpu_weight_pyr_gauss_vec_[gpu_feed_idx_][i];
            cv::cuda::GpuMat _dst_band_weights = gpu_dst_band_weights_[i](rc);

            using namespace cv::cuda::device::blend;

            if (weight_type_ == CV_32F)
            {
                addSrcWeightGpu32F(_src_pyr_laplace, _weight_pyr_gauss, _dst_pyr_laplace, _dst_band_weights, rc);
            }
            else
            {
                addSrcWeightGpu16S(_src_pyr_laplace, _weight_pyr_gauss, _dst_pyr_laplace, _dst_band_weights, rc);
            }
            x_tl /= 2; y_tl /= 2;
            x_br /= 2; y_br /= 2;
        }
        ++gpu_feed_idx_;
        return;
    }
#endif

    // Create the source image Laplacian pyramid
    cv::UMat img_with_border;
    copyMakeBorder(_img, img_with_border, top, bottom, left, right,
                   cv::BORDER_REFLECT);
#if ENABLE_LOG
    LOGLN("  Add border to the source image, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
#endif
#if ENABLE_LOG
    t = getTickCount();
#endif

    std::vector<cv::UMat> src_pyr_laplace;
    cv::detail::createLaplacePyr(img_with_border, num_bands_, src_pyr_laplace);

#if ENABLE_LOG
    LOGLN("  Create the source image Laplacian pyramid, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
#endif
#if ENABLE_LOG
    t = getTickCount();
#endif

    // Create the weight map Gaussian pyramid
    cv::UMat weight_map;
    std::vector<cv::UMat> weight_pyr_gauss(num_bands_ + 1);

    if (weight_type_ == CV_32F)
    {
        mask.getUMat().convertTo(weight_map, CV_32F, 1./255.);
    }
    else // weight_type_ == CV_16S
    {
        mask.getUMat().convertTo(weight_map, CV_16S);
        cv::UMat add_mask;
        compare(mask, 0, add_mask, cv::CMP_NE);
        add(weight_map, cv::Scalar::all(1), weight_map, add_mask);
    }

    copyMakeBorder(weight_map, weight_pyr_gauss[0], top, bottom, left, right, cv::BORDER_CONSTANT);

    for (int i = 0; i < num_bands_; ++i)
        pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

#if ENABLE_LOG
    LOGLN("  Create the weight map Gaussian pyramid, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
#endif
#if ENABLE_LOG
    t = getTickCount();
#endif

    int y_tl = tl_new.y - dst_roi_.y;
    int y_br = br_new.y - dst_roi_.y;
    int x_tl = tl_new.x - dst_roi_.x;
    int x_br = br_new.x - dst_roi_.x;

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    for (int i = 0; i <= num_bands_; ++i)
    {
        cv::Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl);
#ifdef HAVE_OPENCL
        if ( !cv::ocl::isOpenCLActivated() ||
             !ocl_MultiBandBlender_feed(src_pyr_laplace[i], weight_pyr_gauss[i],
                    dst_pyr_laplace_[i](rc), dst_band_weights_[i](rc)) )
#endif
        {
            cv::Mat _src_pyr_laplace = src_pyr_laplace[i].getMat(cv::ACCESS_READ);
            cv::Mat _dst_pyr_laplace = dst_pyr_laplace_[i](rc).getMat(cv::ACCESS_RW);
            cv::Mat _weight_pyr_gauss = weight_pyr_gauss[i].getMat(cv::ACCESS_READ);
            cv::Mat _dst_band_weights = dst_band_weights_[i](rc).getMat(cv::ACCESS_RW);
            if (weight_type_ == CV_32F)
            {
                for (int y = 0; y < rc.height; ++y)
                {
                    const cv::Point3_<short>* src_row = _src_pyr_laplace.ptr<cv::Point3_<short> >(y);
                    cv::Point3_<short>* dst_row = _dst_pyr_laplace.ptr<cv::Point3_<short> >(y);
                    const float* weight_row = _weight_pyr_gauss.ptr<float>(y);
                    float* dst_weight_row = _dst_band_weights.ptr<float>(y);

                    for (int x = 0; x < rc.width; ++x)
                    {
                        dst_row[x].x += static_cast<short>(src_row[x].x * weight_row[x]);
                        dst_row[x].y += static_cast<short>(src_row[x].y * weight_row[x]);
                        dst_row[x].z += static_cast<short>(src_row[x].z * weight_row[x]);
                        dst_weight_row[x] += weight_row[x];
                    }
                }
            }
            else // weight_type_ == CV_16S
            {
                for (int y = 0; y < y_br - y_tl; ++y)
                {
                    const cv::Point3_<short>* src_row = _src_pyr_laplace.ptr<cv::Point3_<short> >(y);
                    cv::Point3_<short>* dst_row = _dst_pyr_laplace.ptr<cv::Point3_<short> >(y);
                    const short* weight_row = _weight_pyr_gauss.ptr<short>(y);
                    short* dst_weight_row = _dst_band_weights.ptr<short>(y);

                    for (int x = 0; x < x_br - x_tl; ++x)
                    {
                        dst_row[x].x += short((src_row[x].x * weight_row[x]) >> 8);
                        dst_row[x].y += short((src_row[x].y * weight_row[x]) >> 8);
                        dst_row[x].z += short((src_row[x].z * weight_row[x]) >> 8);
                        dst_weight_row[x] += weight_row[x];
                    }
                }
            }
        }
#ifdef HAVE_OPENCL
        else
        {
            CV_IMPL_ADD(CV_IMPL_OCL);
        }
#endif

        x_tl /= 2; y_tl /= 2;
        x_br /= 2; y_br /= 2;
    }

#if ENABLE_LOG
    LOGLN("  Add weighted layer of the source image to the final Laplacian pyramid layer, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
#endif
}

void MyBlend::blend(CV_IN_OUT cv::InputOutputArray dst, CV_IN_OUT cv::InputOutputArray dst_mask)
{
    cv::Rect dst_rc(0, 0, dst_roi_final_.width, dst_roi_final_.height);
#if defined(HAVE_CUDA) && defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    if (can_use_gpu_)
    {
        if (!gpu_initialized_)
        {
            gpu_ups_.push_back(std::vector<cv::cuda::GpuMat>(num_bands_+1));
        }

        for (int i = 0; i <= num_bands_; ++i)
        {
            cv::cuda::GpuMat dst_i = gpu_dst_pyr_laplace_[i];
            cv::cuda::GpuMat weight_i = gpu_dst_band_weights_[i];

            using namespace ::cv::cuda::device::blend;
            if (weight_type_ == CV_32F)
            {
                normalizeUsingWeightMapGpu32F(weight_i, dst_i, weight_i.cols, weight_i.rows);
            }
            else
            {
                normalizeUsingWeightMapGpu16S(weight_i, dst_i, weight_i.cols, weight_i.rows);
            }
        }

        // Restore image from Laplacian pyramid
        for (size_t i = num_bands_; i > 0; --i)
        {
            cv::cuda::pyrUp(gpu_dst_pyr_laplace_[i], gpu_ups_[gpu_ups_.size()-1][num_bands_-i]);
            cv::cuda::add(gpu_ups_[gpu_ups_.size()-1][num_bands_-i],
                      gpu_dst_pyr_laplace_[i - 1],
                      gpu_dst_pyr_laplace_[i - 1]);
        }

        // If dst is GpuMat do masking on gpu and return dst as a GpuMat
        // else download the image to cpu and return it as an ordinary Mat
        if (dst.isGpuMat())
        {
            cv::cuda::GpuMat &gpu_dst = dst.getGpuMatRef();

            cv::cuda::compare(gpu_dst_band_weights_[0](dst_rc), WEIGHT_EPS, gpu_dst_mask_, cv::CMP_GT);

            cv::cuda::compare(gpu_dst_mask_, 0, gpu_mask_, cv::CMP_EQ);

            gpu_dst_pyr_laplace_[0](dst_rc).setTo(cv::Scalar::all(0), gpu_mask_);
            gpu_dst_pyr_laplace_[0](dst_rc).convertTo(gpu_dst, CV_16S);

        }
        else
        {
            gpu_dst_pyr_laplace_[0](dst_rc).download(dst_);
            cv::Mat dst_band_weights_0;
            gpu_dst_band_weights_[0].download(dst_band_weights_0);

            compare(dst_band_weights_0(dst_rc), WEIGHT_EPS, dst_mask_, cv::CMP_GT);
            Blender::blend(dst, dst_mask);
        }

        // Set destination Mats to 0 so new image can be blended
        for (size_t i = 0; i < (size_t)(num_bands_ + 1); ++i)
        {
            gpu_dst_band_weights_[i].setTo(0);
            gpu_dst_pyr_laplace_[i].setTo(cv::Scalar::all(0));
        }
        gpu_feed_idx_ = 0;
        gpu_initialized_ = true;
    }
    else
#endif
    {
        cv::UMat dst_band_weights_0;

        for (int i = 0; i <= num_bands_; ++i) {
            cv::detail::normalizeUsingWeightMap(dst_band_weights_[i], dst_pyr_laplace_[i]);
        }

        cv::detail::restoreImageFromLaplacePyr(dst_pyr_laplace_);

        dst_ = dst_pyr_laplace_[0](dst_rc);
        dst_band_weights_0 = dst_band_weights_[0];

        dst_pyr_laplace_.clear();
        dst_band_weights_.clear();

        compare(dst_band_weights_0(dst_rc), WEIGHT_EPS, dst_mask_, cv::CMP_GT);

        Blender::blend(dst, dst_mask);
    }
}
