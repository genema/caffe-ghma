/*
* @Author: gehuama
* @Date:   2017-12-16 15:51:52
* @Last Modified by:   gehuama
* @Last Modified time: 2017-12-16 16:24:47
*/
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L1normLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "L1norm Loss : Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void L1normLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype sum = caffe_cpu_asum(count, diff_.cpu_data());
  Dtype loss = sum / bottom[0]->num();// 1/2N * SSE(y, ygt)
  top[0]->mutable_gpu_data()[0] = loss;
}

template <typename Dtype>
void L1normLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_add_scalar(
	      	bottom[i]->count(),
	      	Dtype(1),
	      	bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L1normLossLayer);

}  // namespace caffe
