/*
* @Author: gehuama
* @Date:   2017-12-09 18:35:17
* @Last Modified by:   gehuama
* @Last Modified time: 2017-12-16 15:34:26
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
  x_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ClarityLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_abs(
    count,
    diff_.cpu_data(),
    diff_.mutable_cpu_data());
  //caffe_copy(count, bottom[2]->cpu_data(), x_.mutable_cpu_data()); // num is the N in each batch
  //caffe_mul(count, x_.cpu_data(), x_.cpu_data(), x_.mutable_cpu_data());
  //caffe_add_scalar(count, Dtype(0.000001), x_.mutable_cpu_data());
  //caffe_sqrt(count, x_.cpu_data(), x_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), x_.cpu_data());
  Dtype loss = sqrt(dot^2 + 0.01^2) / bottom[0]->num(); 
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ClarityLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
        const Dtype sign = (i == 0) ? 1 : -1;
        const Dtype alpha = 1 / (sign * top[0]->cpu_diff()[0]) / bottom[i]->num(); //(const Dtype*)diff_->cpu_data();
        caffe_cpu_axpby(
            bottom[i]->count(),              // count
            alpha,                           // alpha
            diff_.cpu_data(),           // x
            Dtype(0),                        // beta
            bottom[i]->mutable_cpu_diff());  // y
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ClarityLossLayer);
#endif

INSTANTIATE_CLASS(ClarityLossLayer);
REGISTER_LAYER_CLASS(ClarityLoss);

}  // namespace caffe
