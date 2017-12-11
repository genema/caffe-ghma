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
void ClarityLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  //Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype *inputx = bottom[2]->gpu_data();
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), inputx, &dot);
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
          bottom[2]->cpu_data(),                          // a
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ClarityLossLayer);

}  // namespace caffe
