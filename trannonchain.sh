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

srclang="CZ"
explang="FR"

suff=

# if [[ -z ${srclang:3} ]]; then
#   src_mdl=exp/chain/${srclang}${suff}/final.mdl
# else
#   src_mdl=exp/chain/trans/${srclang}/final.mdl
# fi


adapt_lr=1.0


affix=
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=false  
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
num_jobs_initial=2
num_jobs_final=2
minibatch_size=128
relu_dim=576
frames_per_eg=150
remove_egs=true
common_egs_dir=
xent_regularize=0.1



# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [[ -z ${srclang:3} ]]; then
  src_mdl=exp/nnet3/tdnn/${srclang}${suff}/final.mdl
else
  src_mdl=exp/nnet3/trans/${srclang}/final.mdl
fi

dir=exp/nnet3/trans${suff}/${srclang}-${explang}
egs_dir="exp/nnet3/trans/${explang}-egs"

[ -d "$egs_dir" ] && common_egs_dir="$egs_dir"

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

dir=${dir}${affix:+_$affix}$suffix
train_set=${explang}/train$suffix
ali_dir=exp/${explang}/tri3_ali$suffix
lang=data/${explang}/lang


# if we are using the speed-perturbed data we need to generate
# alignments for it.
# local/nnet3/run_ivector_common.sh --stage $stage \
#   --speed-perturb $speed_perturb \
#   --generate-alignments $speed_perturb || exit 1;

# if [ $stage -le 10 ]; then
#   # Create a version of the lang/ directory that has one state per phone in the
#   # topo file. [note, it really has two states.. the first one is only repeated
#   # once, the second one has zero or more repeats.]
#   rm -rf $lang
#   cp -r data/${explang}/lang $lang
#   silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
#   nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
#   # Use our special topology... note that later on may have to tune this
#   # topology.
#   steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
# fi

# if [ $stage -le 11 ]; then
#   # Build a tree using our new topology.
#   steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
#       --leftmost-questions-truncate $leftmost_questions_truncate \
#       --cmd "$train_cmd" 9000 data/$train_set $lang $ali_dir $treedir
# fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs";

  mkdir -p $dir
  num_targets=$(tree-info exp/nnet3/tdnn/${explang}/tree | grep num-pdfs | awk '{print $2}')
  #nnet3-am-copy --edits="remove-output-nodes name=output; remove-orphan-components" $src_mdl $dir/transf.mdl

  mkdir -p $dir/configs
    cat <<EOF > $dir/configs/network.xconfig
  ## adding the layers for chain branch
  # relu-batchnorm-layer name=tdnn7 dim=576 input=Append(tdnn6.batchnorm@-3,tdnn6.batchnorm)
  output-layer name=output-$explang input=tdnn6.batchnorm dim=$num_targets max-change=1.5 presoftmax-scale-file=$dir/configs/presoftmax_prior_scale.vec
EOF
    mkdir -p configs
    cp exp/nnet3/tdnn/${explang}/configs/presoftmax_prior_scale.vec $dir/configs/presoftmax_prior_scale.vec
    local/xconf.py --existing-model $src_mdl \
		   --xconfig-file $dir/configs/network.xconfig \
		   --config-dir $dir/configs
    $train_cmd $dir/log/generate_input_mdl.log \
	   nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$adapt_lr" $src_mdl - \| \
	   nnet3-init - $dir/configs/final.config $dir/configs/temp.raw || exit 1;
    
    nnet3-copy --edits="remove-output-nodes name=output;remove-orphans;rename-node old-name=output-${explang} new-name=output" \
	       $dir/configs/temp.raw  \
	       $dir/input.raw

    copy-transition-model exp/nnet3/tdnn/${explang}/0.mdl - | nnet3-am-init - $dir/input.raw $dir/input.mdl
    # CHECK: AM initialization in train_dnn
fi



if [ $stage -le 13 ]; then
   # nnet3-am-copy --raw=true --edits="set-learning-rate-factor name=* learning-rate-factor=$adapt_lr; set-learning-rate-factor name=output* learning-rate-factor=1.0" \
   # 		$src_mdl $dir/input.raw || exit 1;

# mkdir -p $dir/egs
 # touch $dir/egs/.nodelete # keep egs around when that run dies.

  steps/nnet3/train_dnn.py --stage=$train_stage \
			   --cmd="$decode_cmd" \
			   --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
			   --trainer.num-epochs 2 \
			   --trainer.input-model $dir/input.mdl \
			   --trainer.optimization.num-jobs-initial $num_jobs_initial \
			   --trainer.optimization.num-jobs-final $num_jobs_final \
			   --trainer.optimization.initial-effective-lrate 0.004 \
			   --trainer.optimization.final-effective-lrate 0.00017 \
			   --egs.dir "$common_egs_dir" \
			   --cleanup.preserve-model-interval 100 \
			   --use-gpu true \
			   --feat-dir=data/${train_set}_hires \
			   --ali-dir $ali_dir \
			   --lang data/${explang}/lang \
			   --dir=$dir  || exit 1;

fi


[ ! -d "$egs_dir" ] && mkdir -p "$egs_dir" && cp -r ${dir}/egs ${egs_dir} && rm -rf ${dir}/egs/

graph_dir=exp/${explang}/tri3/graph_tgpr_sri

if [ $stage -le 14 ]; then
  for decode_set in dev eval; do
    (
      steps/nnet3/decode.sh --nj 5 --cmd "$decode_cmd" \
	 $graph_dir data/${explang}/${decode_set}_hires $dir/decode_${decode_set}__tgpr_sri || exit 1;
    ) &
  done
fi
wait;
exit 0;
