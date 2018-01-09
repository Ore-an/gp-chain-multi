#!/bin/bash

# d is as c, but with one extra layer.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

# note: the last column is a version of tdnn_d that was done after the
# changes for the 5.1 version of Kaldi (variable minibatch-sizes, etc.)
# System                  tdnn_c   tdnn_d       tdnn_d[repeat]
# WER on train_dev(tg)      17.37     16.72      16.51
# WER on train_dev(fg)      15.94     15.31      15.34
# WER on eval2000(tg)        20.0      19.2        19.2
# WER on eval2000(fg)        18.2      17.8       17.7
# Final train prob       -1.43781  -1.22859      -1.22215
# Final valid prob       -1.56895    -1.354     -1.31647

stage=0
affix=
train_stage=-10
common_egs_dir=
reporting_email=
remove_egs=true
explang="SP"


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
    cat <<EOF
EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

dir=exp/nnet3/tdnn/${explang}
train_set=${explang}/train
ali_dir=exp/${explang}/tri3_ali
lang=data/${explang}/lang

if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $ali_dir/tree | grep num-pdfs | awk '{print $2}')

  mkdir -p $dir/configs
    cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=576
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=576
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=576
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=576
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=576
  relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=576

  output-layer name=output input=tdnn6 dim=$num_targets max-change=1.5 presoftmax-scale-file=$dir/configs/presoftmax_prior_scale.vec
EOF

    steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi



if [ $stage -le 10 ]; then

  steps/nnet3/train_dnn.py --stage=$train_stage \
			   --cmd="$decode_cmd" \
			   --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
			   --trainer.num-epochs 2 \
			   --trainer.optimization.num-jobs-initial 1 \
			   --trainer.optimization.num-jobs-final 2 \
			   --trainer.optimization.initial-effective-lrate 0.004 \
			   --trainer.optimization.final-effective-lrate 0.00017 \
			   --egs.dir "$common_egs_dir" \
			   --cleanup.remove-egs $remove_egs \
			   --cleanup.preserve-model-interval 100 \
			   --use-gpu true \
			   --feat-dir=data/${train_set}_hires \
			   --ali-dir $ali_dir \
			   --lang data/${explang}/lang \
			   --dir=$dir  || exit 1;

fi

graph_dir=exp/${explang}/tri3/graph_tgpr_sri
if [ $stage -le 11 ]; then
  for decode_set in dev eval; do
    (
      steps/nnet3/decode.sh --nj 5 --cmd "$decode_cmd" \
	 $graph_dir data/${explang}/${decode_set}_hires $dir/decode_${decode_set}__tgpr_sri || exit 1;
    ) &
  done
fi
wait;
exit 0;
