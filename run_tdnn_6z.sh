#!/bin/bash

# 6z is as 6y, but fixing the right-tolerance in the scripts to default to 5 (as
# the default is in the code), rather than the previous script default value of
# 10 which I seem to have added to the script around Feb 9th.
# definitely better than 6y- not clear if we have managed to get the same
# results as 6v (could indicate that the larger frames-per-iter is not helpful?
# but I'd rather not decrease it as it would hurt speed).

# local/chain/compare_wer.sh 6v 6y 6z
# System                       6v        6y        6z
# WER on train_dev(tg)      15.00     15.36     15.18
# WER on train_dev(fg)      13.91     14.19     14.06
# WER on eval2000(tg)        17.2      17.2      17.2
# WER on eval2000(fg)        15.7      15.8      15.6
# Final train prob      -0.105012 -0.102139 -0.106268
# Final valid prob      -0.125877 -0.119654 -0.126726
# Final train prob (xent)      -1.54736  -1.55598  -1.4556
# Final valid prob (xent)      -1.57475  -1.58821  -1.50136

# 6y is as 6w, but after fixing the config-generation script to use
# a higher learning-rate factor for the final xent layer (it was otherwise
# training too slowly).

# 6w is as 6v (a new tdnn-based recipe), but using 1.5 million not 1.2 million
# frames per iter (and of course re-dumping the egs).

# this is same as v2 script but with xent-regularization
# it has a different splicing configuration
set -e

# configs for 'chain'

explang="WSJ"

affix=
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=false
dir=exp/chain/ # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=

# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
# smoothing options
self_repair_scale=0.00001
# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=1
num_jobs_final=2
minibatch_size=128
relu_dim=576
frames_per_eg=150
remove_egs=false
common_egs_dir=
xent_regularize=0.1



# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}/${explang}${affix:+_$affix}$suffix
train_set=${explang}/train$suffix
ali_dir=exp/${explang}/tri3_ali$suffix
treedir=exp/chain/${explang}-tree$suffix
lang=data/${explang}/lang_chain


# if we are using the speed-perturbed data we need to generate
# alignments for it.
# local/nnet3/run_ivector_common.sh --stage $stage \
#   --speed-perturb $speed_perturb \
#   --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/${explang}/tri3_ali$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/${explang}/lang exp/${explang}/tri3 exp/${explang}/tri3_lats$suffix
  rm exp/${explang}/tri3_lats$suffix/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/${explang}/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 9000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs";
  # if [ ! -z "$relu_dim" ]; then
  #   dim_opts="--relu-dim $relu_dim"
  # else
  #   dim_opts="--pnorm-input-dim $pnorm_input_dim --pnorm-output-dim  $pnorm_output_dim"
  # fi

  # # create the config files for nnet initialization
  # repair_opts=${self_repair_scale:+" --self-repair-scale-nonlinearity $self_repair_scale "}

  # steps/nnet3/tdnn/make_configs.py \
  #   $repair_opts \
  #   --feat-dir data/${train_set} \
  #   --tree-dir $treedir \
  #   $dim_opts \
  #   --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0" \
  #   --use-presoftmax-prior-scale false \
  #   --xent-regularize $xent_regularize \
  #   --xent-separate-forward-affine true \
  #   --include-log-softmax false \
  #   --final-layer-normalize-target $final_layer_normalize_target \
  #   $dir/configs || exit 1;
  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
    cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=512
  relu-batchnorm-layer name=tdnn2 dim=512 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn3 dim=512 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn4 dim=512 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn5 dim=512 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn6 dim=512 input=Append(-6,-3,0)
  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain dim=512 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5
  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=512 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
    steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs  
fi



if [ $stage -le 13 ]; then
 mkdir -p $dir/egs
 touch $dir/egs/.nodelete # keep egs around when that run dies.
 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/${explang}/tri3_lats$suffix \
    --dir $dir  || exit 1;

 nnet3-am-init $dir/0.trans_mdl $dir/final.raw $dir/final.mdl
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/${explang}/lang_test_tgpr_sri $dir $dir/graph_tgpr_sri
fi

decode_suff=_tgpr_sri
graph_dir=$dir/graph_tgpr_sri
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 5 --cmd "$decode_cmd" $iter_opts \
          $graph_dir data/${explang}/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
      ) &
  done
fi

wait;

# if [ $stage -le 16 ]; then
#   for decode_set in dev eval; do
#     (
#       steps/score_kaldi.sh --cmd "$decode_cmd" data/${explang}/${decode_set}_hires \
# 			   $graph_dir $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff}
#     ) &
#   done
# fi

wait;
exit 0;
