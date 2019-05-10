#pragma once

//pytorch
#include <torch/torch.h>

//eigen
#include <Eigen/Dense>

//c++
#include <iostream>

//my stuff
#include "surfel_renderer/utils/MiscUtils.h"

//opencv
#include "opencv2/opencv.hpp"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp> //needs to be added after torch.h otherwise loguru stops printing for some reason

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXfRowMajor;

//grabs a cv mat in whatever type or nr of channels it has and returns a tensor of shape NCHW. Also converts from BGR to RGB
inline torch::Tensor mat2tensor(const cv::Mat& cv_mat_bgr){
    CHECK( cv_mat_bgr.isContinuous()) << "cv_mat should be continuous in memory because we will wrap it directly.";

    cv::Mat cv_mat;
    if(cv_mat_bgr.channels()==3){
        cvtColor(cv_mat_bgr, cv_mat, cv::COLOR_BGR2RGB);
    }else if(cv_mat_bgr.channels()==4){
        cvtColor(cv_mat_bgr, cv_mat, cv::COLOR_BGRA2RGBA);
    }else{
        cv_mat=cv_mat_bgr;
    }

    //get the scalar type of the tensor, the types supported by torch are here https://github.com/pytorch/pytorch/blob/1a742075ee97b9603001188eeec9c30c3fe8a161/torch/csrc/utils/python_scalars.h
    at::ScalarType tensor_scalar_type; 
    unsigned char cv_mat_type=er::utils::type2byteDepth(cv_mat.type());
    if(cv_mat_type==CV_8U ){
        tensor_scalar_type=at::kByte;
    }else if(cv_mat_type==CV_32S ){
        tensor_scalar_type=at::kInt;
    }else if(cv_mat_type==CV_32F ){
        tensor_scalar_type=at::kFloat;
    }else if(cv_mat_type==CV_64F ){
        tensor_scalar_type=at::kDouble;
    }else{
        LOG(FATAL) << "Not a supported type of cv mat";
    }


    int c=cv_mat.channels();
    int h=cv_mat.rows;
    int w=cv_mat.cols;

    // torch::Tensor wrapped_mat = torch::from_blob(cv_mat.data,  /*sizes=*/{ 1, 3, 512, 512 }, tensor_scalar_type);
    torch::Tensor wrapped_mat = torch::from_blob(cv_mat.data,  /*sizes=*/{ 1, h, w, c }, tensor_scalar_type); //opencv stores the mat in hwc format where c is the fastest changing and h is the slowest

    torch::Tensor tensor = wrapped_mat.clone(); //we have to take ownership of the data of the cv_mat, otherwise the cv_mat might go out of scope and then we will point to undefined data

    //we want a tensor oh nchw instead of nhwc
    tensor = tensor.permute({0, 3, 1, 2}); //the order in which we declared then dimensions is 0,1,2,3 and they go like 1,h,w,c. With this permute we put them in 0,3,1,2 so in 1,c,h,w

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

    torch::Tensor tensor=tensor_in.to(at::kCPU);

    // tensor = tensor.permute({0, 2, 3, 1}); //go from 1chw to 1hwc which is what opencv likes

    //get type of tensor
    at::ScalarType tensor_scalar_type; 
    tensor_scalar_type=tensor.scalar_type();
    int c=tensor.size(1);
    // std::cout << "nr channels is " << c <<std::endl;
    //infer a good type for the cv mat and read the channels of the cv mat
    std::vector<cv::Mat> channels;
    int cv_mat_type;
    cv::Mat final_mat;
    if(c==1){
        //we get only the one channel, we have to clone them because we want direct access to their memory so we can copy it into a mat channel , but slice by itself doesnt change the memory, it mearly gives a view into it
        auto red_output = tensor.slice(1, 0, 1).clone(); //along dimensiom 1 (corresponding to the channels) go from 0,1 and get all the data along there

        if(tensor_scalar_type==at::kByte ){
            cv_mat_type=CV_8UC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<uint8_t>()) );
        }else if(tensor_scalar_type==at::kInt ){
            cv_mat_type=CV_32SC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<int32_t>()) );
        }else if(tensor_scalar_type==at::kFloat ){
            cv_mat_type=CV_32FC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<float>()) );
        }else if(tensor_scalar_type==at::kDouble ){
            cv_mat_type=CV_64FC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<double>()) );
        }else{
            LOG(FATAL) << "Not a supported type of tensor_type";
        }

        //we merge the channels and return the final image
        cv::merge(channels, final_mat);
        return final_mat;

    }else if (c==2){
        //we get two channels, we have to clone them because we want direct access to their memory so we can copy it into a mat channel, but slice by itself doesnt change the memory, it mearly gives a view into it
        auto red_output = tensor.slice(1, 0, 1).clone(); //along dimensiom 1 (corresponding to the channels) go from 0,1 and get all the data along there
        auto green_output = tensor.slice(1, 1, 2).clone();

        if(tensor_scalar_type==at::kByte ){
            cv_mat_type=CV_8UC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<uint8_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<uint8_t>()) );
        }else if(tensor_scalar_type==at::kInt ){
            cv_mat_type=CV_32SC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<int32_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<int32_t>()) );
        }else if(tensor_scalar_type==at::kFloat ){
            cv_mat_type=CV_32FC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<float>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<float>()) );
        }else if(tensor_scalar_type==at::kDouble ){
            cv_mat_type=CV_64FC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<double>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<double>()) );
        }else{
            LOG(FATAL) << "Not a supported type of tensor_type";
        }

        //we merge the channels and return the final image
        cv::merge(channels, final_mat);
        return final_mat;

    }else if(c==3){
        //we get 3 channels , we have to clone them because we want direct access to their memory so we can copy it into a mat channel, but slice by itself doesnt change the memory, it mearly gives a view into it
        auto red_output = tensor.slice(1, 0, 1).clone(); //along dimensiom 1 (corresponding to the channels) go from 0,1 and get all the data along there
        auto green_output = tensor.slice(1, 1, 2).clone();
        auto blue_output = tensor.slice(1, 2, 3).clone();
        // std::cout << "red_output is "<< red_output << std::endl;
        // std::cout << "green_output is "<< green_output << std::endl;
        // std::cout << "blue_output is "<< blue_output << std::endl;

        if(tensor_scalar_type==at::kByte ){
            cv_mat_type=CV_8UC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<uint8_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<uint8_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, blue_output.data<uint8_t>()) );
        }else if(tensor_scalar_type==at::kInt ){
            std::cout << "int" << std::endl;
            cv_mat_type=CV_32SC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<int32_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<int32_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, blue_output.data<int32_t>()) );
        }else if(tensor_scalar_type==at::kFloat ){
            cv_mat_type=CV_32FC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<float>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<float>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, blue_output.data<float>()) );
        }else if(tensor_scalar_type==at::kDouble ){
            cv_mat_type=CV_64FC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<double>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<double>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, blue_output.data<double>()) );
        }else{
            LOG(FATAL) << "Not a supported type of tensor_type";
        }

        //we merge the channels and return the final image
        cv::merge(channels, final_mat);
        cv::Mat final_mat_rgb; 
        cvtColor(final_mat, final_mat_rgb, cv::COLOR_BGR2RGB);
        return final_mat_rgb;

        // return final_mat;

    }else if(c==4){
        //we get 4 channels, we have to clone them because we want direct access to their memory so we can copy it into a mat channel, but slice by itself doesnt change the memory, it mearly gives a view into it
        auto red_output = tensor.slice(1, 0, 1).clone(); //along dimensiom 1 (corresponding to the channels) go from 0,1 and get all the data along there
        auto green_output = tensor.slice(1, 1, 2).clone();
        auto blue_output = tensor.slice(1, 2, 3).clone();
        auto alpha_output = tensor.slice(1, 3, 4).clone();

        if(tensor_scalar_type==at::kByte ){
            cv_mat_type=CV_8UC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<uint8_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<uint8_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, blue_output.data<uint8_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, alpha_output.data<uint8_t>()) );
        }else if(tensor_scalar_type==at::kInt ){
            cv_mat_type=CV_32SC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<int32_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<int32_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, blue_output.data<int32_t>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, alpha_output.data<int32_t>()) );
        }else if(tensor_scalar_type==at::kFloat ){
            cv_mat_type=CV_32FC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<float>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<float>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, blue_output.data<float>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, alpha_output.data<float>()) );
        }else if(tensor_scalar_type==at::kDouble ){
            cv_mat_type=CV_64FC1;
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, red_output.data<double>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, green_output.data<double>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, blue_output.data<double>()) );
            channels.push_back( cv::Mat(tensor.size(2), tensor.size(3), cv_mat_type, alpha_output.data<double>()) );
        }else{
            LOG(FATAL) << "Not a supported type of tensor_type";
        }

        //we merge the channels and return the final image
        cv::merge(channels, final_mat);
        cv::Mat final_mat_rgba; 
        cvtColor(final_mat, final_mat_rgba, cv::COLOR_BGRA2RGBA);
        return final_mat_rgba;

    }else{
        LOG(FATAL) << "Not a supported number of channels. c is " << c;
    }

    
}

//converts a RowMajor eigen matrix of size HW into a tensor of size 1HW
inline torch::Tensor eigen2tensor(const EigenMatrixXfRowMajor& eigen_mat){

    torch::Tensor wrapped_mat = torch::from_blob(const_cast<float*>(eigen_mat.data()),  /*sizes=*/{ 1, eigen_mat.rows(), eigen_mat.cols() }, at::kFloat); 
    torch::Tensor tensor = wrapped_mat.clone(); //we have to take ownership of the data, otherwise the eigen_mat might go out of scope and then we will point to undefined data

    return tensor;
    
}

//converts tensor of shape 1hw into a RowMajor eigen matrix of size HW 
inline EigenMatrixXfRowMajor tensor2eigen(const torch::Tensor& tensor_in){

    torch::Tensor tensor=tensor_in.to(at::kCPU);

    int rows=tensor.size(1);
    int cols=tensor.size(2);

    EigenMatrixXfRowMajor eigen_mat(rows,cols);
    eigen_mat=Eigen::Map<EigenMatrixXfRowMajor> (tensor.data<float>(),rows,cols);

    //make a deep copy of it because map does not actually take ownership
    EigenMatrixXfRowMajor eigen_mat_copy;
    eigen_mat_copy=eigen_mat;

    return eigen_mat_copy;
    
}


