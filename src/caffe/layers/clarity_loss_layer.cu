/*
* @Author: gehuama
* @Date:   2017-12-09 18:35:17
* @Last Modified by:   gehuama
* @Last Modified time: 2017-12-09 18:35:17
*/

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "math.h"

namespace caffe {

template <typename Dtype>
void ClarityLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << " CLARITY LOSS --> Inputs bottom[0] [1] must have the same dim.";
  CHECK_EQ(bottom[0]->count(1), bottom[2]->count(1))
      << " CLARITY LOSS --> Inputs bottom[0] [2] must have the same dim.";
  diff_.ReshapeLike(*bottom[0]);
  x_.ReshapeLike(*bottom[0]);
  x2_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ClarityLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  //caffe_copy(count, bottom[2]->gpu_data(), x_.mutable_gpu_data()); // num is the N in each batch
  //caffe_gpu_sqrt(count, x_.gpu_data(), x_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  //caffe_gpu_add_scalar(count, Dtype(0.000001), x_.mutable_gpu_data());
  Dtype loss = sqrt(dot^2+0.01^2) / bottom[0]->num(); 
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ClarityLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    //printf(" propagate_down=%d, i=%d",propagate_down[i], i);
    if (propagate_down[i]) {
        const Dtype sign = (i == 0) ? 1 : -1;
        const Dtype alpha = 1 / (sign * top[0]->cpu_diff()[0]) / bottom[i]->num(); //(const Dtype*)diff_->cpu_data();
        caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          diff_.gpu_data(),           // x
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // y
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ClarityLossLayer);

}  // namespace caffe
