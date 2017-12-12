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
  x_.ReshaleLike(*bottom[0]);
}

template <typename Dtype>
void ClarityLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  caffe_copy(count, bottom[2]->gpu_data(), x_.mutable_gpu_data()); // num is the N in each batch
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), x_.gpu_data(), &dot);
  Dtype loss                    = dot / bottom[0]->num(); 
  top[0]->mutable_gpu_data()[0] = loss;
}

template <typename Dtype>
void ClarityLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num(); //(const Dtype*)diff_->cpu_data();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          bottom[2]->gpu_data(),                          // a
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ClarityLossLayer);

}  // namespace caffe
