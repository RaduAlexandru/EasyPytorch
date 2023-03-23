#pragma once

//pytorch
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

//eigen
#include <Eigen/Dense>

//c++
#include <iostream>

//my stuff
// #include "surfel_renderer/utils/MiscUtils.h"
#include "opencv_utils.h"

//opencv
#include "opencv2/opencv.hpp"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp> //needs to be added after torch.h otherwise loguru stops printing for some reason

using namespace radu::utils;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXfRowMajor;

//grabs a cv mat in whatever type or nr of channels it has and returns a tensor of shape NCHW. Also converts from BGR to RGB
inline torch::Tensor mat2tensor(const cv::Mat& mat_in, const bool flip_red_blue){

    CHECK(mat_in.data) << "The input mat has no data or is empty";

    //make continous if it's not
    cv::Mat cv_mat_cont;
    if(!mat_in.isContinuous()){
        cv_mat_cont=mat_in.clone(); //cloning makes it continous
    }else{
        cv_mat_cont=mat_in;
    }


    CHECK( cv_mat_cont.isContinuous()) << "cv_mat should be continuous in memory because we will wrap it directly.";

    cv::Mat cv_mat;
    if(cv_mat_cont.channels()==3 && flip_red_blue){
        cvtColor(cv_mat_cont, cv_mat, cv::COLOR_BGR2RGB);
    }else if(cv_mat_cont.channels()==4 && flip_red_blue){
        cvtColor(cv_mat_cont, cv_mat, cv::COLOR_BGRA2RGBA);
    }else{
        cv_mat=cv_mat_cont;
    }

    //get the scalar type of the tensor, the types supported by torch are here https://github.com/pytorch/pytorch/blob/1a742075ee97b9603001188eeec9c30c3fe8a161/torch/csrc/utils/python_scalars.h
    at::ScalarType tensor_scalar_type;
    unsigned char cv_mat_type=type2byteDepth(cv_mat.type());
    if(cv_mat_type==CV_8U ){
        tensor_scalar_type=at::kByte;
    }else if(cv_mat_type==CV_32S ){
        tensor_scalar_type=at::kInt;
    }else if(cv_mat_type==CV_32F ){
        tensor_scalar_type=at::kFloat;
    }else if(cv_mat_type==CV_64F ){
        tensor_scalar_type=at::kDouble;
    }else{
        tensor_scalar_type=at::kFloat; //this line doesn't matter since we do a log(fatal) afterwards but this is here so that the compiler doesn't complain about uninitialzied variable
        LOG(FATAL) << "Not a supported type of cv mat";
    }


    int c=cv_mat.channels();
    int h=cv_mat.rows;
    int w=cv_mat.cols;

    // torch::Tensor wrapped_mat = torch::from_blob(cv_mat.data,  /*sizes=*/{ 1, 3, 512, 512 }, tensor_scalar_type);
    torch::Tensor wrapped_mat = torch::from_blob(cv_mat.data,  /*sizes=*/{ 1, h, w, c }, tensor_scalar_type); //opencv stores the mat in hwc format where c is the fastest changing and h is the slowest

    torch::Tensor tensor = wrapped_mat.clone(); //we have to take ownership of the data of the cv_mat, otherwise the cv_mat might go out of scope and then we will point to undefined data

    //we want a tensor oh nchw instead of nhwc
    tensor = tensor.permute({0, 3, 1, 2}).contiguous(); //the order in which we declared then dimensions is 0,1,2,3 and they go like 1,h,w,c. With this permute we put them in 0,3,1,2 so in 1,c,h,w

    // std::cout << "mat2tensor. output a tensor of size" << tensor.sizes();
    return tensor;

}

//converts a tensor from nchw to a cv mat. Assumes the number of batches N is 1
//most of it is from here https://github.com/jainshobhit/pytorch-cpp-examples/blob/master/libtorch_inference.cpp#L39
inline cv::Mat tensor2mat(const torch::Tensor& tensor_in){
    CHECK(tensor_in.dim()==4) << "The tensor should be a 4D one with shape NCHW, however it has dim: " << tensor_in.dim();
    CHECK(tensor_in.size(0)==1) << "The tensor should have only one batch, so the first dimension should be 1. However the sizes are: " << tensor_in.sizes();
    CHECK(tensor_in.size(1)<=4) << "The tensor should have 1,2,3 or 4 channels so the dimension 3 should be in that range. However the sizes are: " << tensor_in.sizes();

    // std::cout << "tensor2mat. Received a tensor of size" << tensor_in.sizes();

    torch::Tensor tensor=tensor_in.to(at::kCPU).contiguous();

    //get type of tensor
    at::ScalarType tensor_scalar_type;
    tensor_scalar_type=tensor.scalar_type();
    int c=tensor.size(1);

    std::vector<cv::Mat> channels;
    int cv_mat_type=-1;
    cv::Mat final_mat;
    if(tensor_scalar_type==at::kByte ){
        cv_mat_type=CV_8UC1;
    }else if(tensor_scalar_type==at::kInt ){
        cv_mat_type=CV_32SC1;
    }else if(tensor_scalar_type==at::kFloat ){
        cv_mat_type=CV_32FC1;
    }else if(tensor_scalar_type==at::kDouble ){
        cv_mat_type=CV_64FC1;
    }else{
        LOG(FATAL) << "Not a supported type of tensor_type";
    }

    //copy each channel into a different cv mat
    //Using data_ptr we can get the raw ptr to the memory of the tensor. This will give us either a pointer to host memory or device memory. We want it in host so have to copy to cpu
    //we have to clone the channels because we want direct access to their memory so we can copy it into a mat channel , but slice by itself doesnt change the memory, it mearly gives a view into it
    for(int c_idx=0; c_idx<c; c_idx++){
        // auto channel = tensor.slice(1, c_idx, c_idx+1).clone(); //along dimensiom 1 (corresponding to the channels) go from 0,1 and get all the data along there
        auto channel = tensor.slice(1, c_idx, c_idx+1).clone(); //along dimensiom 1 (corresponding to the channels) go from 0,1 and get all the data along there
        channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, channel.data_ptr() ).clone() ); //need to clone because the channel will go out of scope as soon as the loop is done
    }


    //we merge the channels and return the final image
    cv::Mat merged_mat;
    cv::merge(channels, merged_mat);
    // if(c==3){
    //     cvtColor(merged_mat, final_mat, cv::COLOR_BGR2RGB);
    // }else if(c==4){
    //     cvtColor(merged_mat, final_mat, cv::COLOR_BGRA2RGBA);
    // }else{
    //     final_mat=merged_mat;
    // }
    final_mat =merged_mat;
    return final_mat;


}

//converts a RowMajor eigen matrix of size HW into a tensor of size HW
inline torch::Tensor eigen2tensor(const EigenMatrixXfRowMajor& eigen_mat){

    torch::Tensor wrapped_mat = torch::from_blob(const_cast<float*>(eigen_mat.data()),  /*sizes=*/{ eigen_mat.rows(), eigen_mat.cols() }, at::kFloat);
    torch::Tensor tensor = wrapped_mat.clone(); //we have to take ownership of the data, otherwise the eigen_mat might go out of scope and then we will point to undefined data

    return tensor;

}

//converts a RowMajor eigen matrix of size HW cv::Mat of size XY
inline cv::Mat eigen2mat(const EigenMatrixXfRowMajor& eigen_mat, const int rows, const int cols){

    CHECK(eigen_mat.rows()==rows*cols) << "We need a row in the eigen mat for each pixel in the image of the cv mat. However nr of rows in the eigen mat is " << eigen_mat.rows() << " while rows*cols of the cv mat is " <<rows*cols;

    int cv_mat_type=-1;
    if(eigen_mat.cols()==1){
        cv_mat_type=CV_32FC1;
    }else if(eigen_mat.cols()==2){
        cv_mat_type=CV_32FC2;
    }else if(eigen_mat.cols()==3){
        cv_mat_type=CV_32FC3;
    }else if(eigen_mat.cols()==4){
        cv_mat_type=CV_32FC4;
    }

    cv::Mat cv_mat (rows, cols, cv_mat_type, (void*)eigen_mat.data() );

    return cv_mat.clone();

}

//converts tensor of shape hw into a RowMajor eigen matrix of size HW
inline EigenMatrixXfRowMajor tensor2eigen(const torch::Tensor& tensor_in){

    CHECK(tensor_in.dim()==2) << "The tensor should be a 2D one with shape HW, however it has dim: " << tensor_in.dim();
    CHECK(tensor_in.scalar_type()==at::kFloat ) << "Tensor should be float. Didn't have time to write templates for this functions";

    torch::Tensor tensor=tensor_in.to(at::kCPU);

    int rows=tensor.size(0);
    int cols=tensor.size(1);

    EigenMatrixXfRowMajor eigen_mat(rows,cols);
    eigen_mat=Eigen::Map<EigenMatrixXfRowMajor> (tensor.data_ptr<float>(),rows,cols);

    //make a deep copy of it because map does not actually take ownership
    EigenMatrixXfRowMajor eigen_mat_copy;
    eigen_mat_copy=eigen_mat;

    return eigen_mat_copy;

}


// //prints the current cuda mem used, including cached memory. Taken from https://github.com/pytorch/pytorch/issues/17433
// inline void display_c10_cuda_mem_stat(int32_t sleep_time) {
//     printf("currentMemoryAllocated/[maxMemoryAllocated]: \t %0.1f/[%0.1f] MB\n ",
//         c10::cuda::CUDACachingAllocator::currentMemoryAllocated(0) / 1024.0 / 1024.0,
//         c10::cuda::CUDACachingAllocator::maxMemoryAllocated(0) / 1024.0 / 1024.0);
//     printf("currentMemoryCached/[maxMemoryCached]: \t %0.1f/[%0.1f] MB\n",
//         c10::cuda::CUDACachingAllocator::currentMemoryCached(0) / 1024.0 / 1024.0,
//         c10::cuda::CUDACachingAllocator::maxMemoryCached(0) / 1024.0 / 1024.0);
//     std::this_thread::sleep_for(std::chrono::milliseconds(1000*sleep_time));
// }

//converts a std::vector<float> to a one dimensional tensor with whatever size the vector has
inline torch::Tensor vec2tensor(const std::vector<float>& vec){

    torch::Tensor wrapped_mat = torch::from_blob(const_cast<float*>(vec.data() ),  /*sizes=*/{ (int)vec.size() }, at::kFloat);
    torch::Tensor tensor = wrapped_mat.clone(); //we have to take ownership of the data, otherwise the eigen_mat might go out of scope and then we will point to undefined data

    return tensor;

}

//empties cache used by cuda
inline void cuda_clear_cache(){
    c10::cuda::CUDACachingAllocator::emptyCache();
}
