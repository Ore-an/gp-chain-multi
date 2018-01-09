#!/bin/bash

# Copyright 2016 Pegah Ghahremani

# This script can be used for training multilingual setup using different
# languages (specifically babel languages) with no shared phones.
# It will generates separate egs directory for each dataset and combine them
# during training.
# In the new multilingual training setup, mini-batches of data corresponding to
# different languages are randomly combined to generate egs.*.scp files
# using steps/nnet3/multilingual/combine_egs.sh and generated egs.*.scp files used
# for multilingual training.
#
# For all languages, we share all except last hidden layer and there is separate final
# layer per language.
# The bottleneck layer can be added to the network structure using --bnf-dim option
#
# The script requires baseline PLP features and alignment (e.g. tri5_ali) for all languages.
# and it will generate 40dim MFCC + pitch features for all languages.
#
# The global iVector extractor trained using all languages by specifying
# --use-global-ivector-extractor and the iVectors are extracts for all languages.
#
# local.conf should exists (check README.txt), which contains configs for
# multilingual training such as lang_list as array of space-separated languages used
# for multilingual training.
#
echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
set -e

remove_egs=false
cmd=queue.pl
srand=0
stage=0
train_stage=-10
get_egs_stage=-10
decode_stage=-10
num_jobs_initial=2
num_jobs_final=2
speed_perturb=false
use_pitch=false  # if true, pitch feature used to train multilingual setup
use_pitch_ivector=false # if true, pitch feature used in ivector extraction.
use_ivector=false
megs_dir=
alidir=tri3_ali
suffix=
feat_suffix=_hires      # The feature suffix describing features used in
                        # multilingual training
                        # _hires_mfcc -> 40dim MFCC
                        # _hire_mfcc_pitch -> 40dim MFCC + pitch
                        # _hires_mfcc_pitch_bnf -> 40dim MFCC +pitch + BNF
# corpora
# language list used for multilingual training
# The map for lang-name to its abreviation can be find in
# local/prepare_flp_langconf.sh
# e.g lang_list=(101-cantonese 102-assamese 103-bengali)
#lang_list=(101-cantonese 102-assamese 103-bengali)
lang_list=(SP CZ FR GE PO)
lang2weight="1.0,1.0,1.0,1.0,1.0,1.0"

# The language in this list decodes using Hybrid multilingual system.
# e.g. decode_lang_list=(101-cantonese)
decode_lang_list=(SP CZ FR GE PO)

ivector_suffix=  # if ivector_suffix = _gb, the iVector extracted using global iVector extractor
                   # trained on pooled data from all languages.
                   # Otherwise, it uses iVector extracted using local iVector extractor.
bnf_dim=           # If non-empty, the bottleneck layer with this dimension is added at two layers before softmax.
dim=625
use_flp=false      # If true, fullLP training data and configs used for training.
dir=exp/nnet3/multi

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

# [ ! -f local.conf ] && echo 'the file local.conf does not exist! Read README.txt for more details.' && exit 1;
# . local.conf || exit 1;

num_langs=${#lang_list[@]}

echo "$0 $@"  # Print the command line for logging
if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

for lang_index in `seq 0 $[$num_langs-1]`; do
  for f in data/${lang_list[$lang_index]}/train/{feats.scp,text} exp/${lang_list[$lang_index]}/$alidir/ali.1.gz exp/${lang_list[$lang_index]}/$alidir/tree; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
done

if [ "$speed_perturb" == "true" ]; then
  suffix=${suffix}_sp
fi

if $use_pitch; then feat_suffix=${feat_suffix}_pitch ; fi
dir=${dir}${suffix}

for lang_index in `seq 0 $[$num_langs-1]`; do
  multi_data_dirs[$lang_index]=data/${lang_list[$lang_index]}/train${suffix}${feat_suffix}
  multi_egs_dirs[$lang_index]=exp/nnet3/multi/${lang_list[$lang_index]}-egs
  multi_ali_dirs[$lang_index]=exp/${lang_list[$lang_index]}/${alidir}${suffix}
done

ivector_dim=0
feat_dim=`feat-to-dim scp:${multi_data_dirs[0]}/feats.scp -`
set +x
if [ $stage -le 9 ]; then
  echo "$0: creating multilingual neural net configs using the xconfig parser";
  mkdir -p $dir/configs
  ivector_node_xconfig=""
  ivector_to_append=""
  if $use_ivector; then
    ivector_node_xconfig="input dim=$ivector_dim name=ivector"
    ivector_to_append=", ReplaceIndex(ivector, t, 0)"
  fi
  cat <<EOF > $dir/configs/network.xconfig
  $ivector_node_xconfig
  input dim=$feat_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here
  
  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=$dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$dim
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$dim 
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$dim
  relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=$dim
  # adding the layers for diffrent language's output
EOF
  # added separate outptut layer and softmax for all languages.
  echo $num_langs
  for lang_index in `seq 0 $[$num_langs-1]`;do
    num_targets=`tree-info ${multi_ali_dirs[$lang_index]}/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;

    echo " relu-renorm-layer name=prefinal-affine-lang-${lang_index} input=tdnn6 dim=$dim"
    echo " output-layer name=output-${lang_index} dim=$num_targets max-change=1.5"
  done >> $dir/configs/network.xconfig

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/ \
    --nnet-edits="rename-node old-name=output-0 new-name=output"

  cat <<EOF >> $dir/configs/vars
add_lda=false
include_log_softmax=false
EOF

  # removing the extra output node "output-tmp" added for back-compatiblity with
  # xconfig to config conversion.
  nnet3-copy --edits="remove-output-nodes name=output-tmp" $dir/configs/ref.raw $dir/configs/ref.raw || exit 1;
fi

if [ $stage -le 9 ]; then
  echo "$0: Generates separate egs dir per language for multilingual training."
  # sourcing the "vars" below sets
  #model_left_context=(something)
  #model_right_context=(something)
  #num_hidden_layers=(something)
  . $dir/configs/vars || exit 1;
  ivec="${multi_ivector_dirs[@]}"
  if $use_ivector; then
    ivector_opts=(--online-multi-ivector-dirs "$ivec")
  fi
  local/nnet3/prepare_multilingual_egs.sh --cmd "$decode_cmd" \
    "${ivector_opts[@]}" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --left-context $model_left_context --right-context $model_right_context \
    $num_langs ${multi_data_dirs[@]} ${multi_ali_dirs[@]} ${multi_egs_dirs[@]} || exit 1;

fi
wait;
if [ -z $megs_dir ];then
  megs_dir=$dir/egs
fi

if [ $stage -le 10 ] && [ ! -z $megs_dir ]; then
  echo "$0: Generate multilingual egs dir using "
  echo "separate egs dirs for multilingual training."
  common_egs_dir="${multi_egs_dirs[@]} $megs_dir"
  steps/nnet3/multilingual/combine_egs.sh $egs_opts \
    --cmd "$decode_cmd" \
    --samples-per-iter 400000 \
    $num_langs ${common_egs_dir[@]} || exit 1;
fi


if [ $stage -le 11 ]; then
  common_ivec_dir=
  if $use_ivector;then
    common_ivec_dir=${multi_ivector_dirs[0]}
  fi
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.minibatch-size=256,128 \
    --trainer.samples-per-iter=400000 \
    --trainer.max-param-change=2.0 \
    --trainer.srand=$srand \
    --feat-dir ${multi_data_dirs[0]} \
    --feat.online-ivector-dir "$common_ivec_dir" \
    --egs.dir $megs_dir \
    --use-dense-targets false \
    --targets-scp ${multi_ali_dirs[0]} \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --use-gpu true \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  for lang_index in `seq 0 $[$num_langs-1]`;do
    lang_dir=$dir/${lang_list[$lang_index]}
    mkdir -p  $lang_dir
    echo "$0: rename output name for each lang to 'output' and "
    echo "add transition model."
    nnet3-copy --edits="rename-node old-name=output-$lang_index new-name=output;remove-output-nodes name=output-*; remove-orphans;" \
      $dir/final.raw - | \
      nnet3-am-init ${multi_ali_dirs[$lang_index]}/final.mdl - \
      $lang_dir/final.mdl || exit 1;
    cp $dir/cmvn_opts $lang_dir/cmvn_opts || exit 1;
    echo "$0: compute average posterior and readjust priors for language ${lang_list[$lang_index]}."
    steps/nnet3/adjust_priors.sh --cmd "$decode_cmd" \
				 --use-gpu true \
				 --iter final --use-raw-nnet false --use-gpu true \
				 $lang_dir ${multi_egs_dirs[$lang_index]} || exit 1;
  done
fi

# decoding different languages

if [ $stage -le 13 ]; then
  num_decode_lang=${#decode_lang_list[@]}
  for lang_index in `seq 0 $[$num_decode_lang-1]`; do
    graph_dir=exp/${decode_lang_list[$lang_index]}/tri3/graph_tgpr_sri
    for decode_set in dev eval; do
      (
	if [ ! -f $dir/${decode_lang_list[$lang_index]}/decode_${decode_set}__tgpr_sri/.done ]; then
	  echo "Decoding lang ${decode_lang_list[$lang_index]} using multilingual hybrid model $dir"
	  steps/nnet3/decode.sh --nj 5 --cmd "$decode_cmd" --iter final_adj \
				$graph_dir data/${decode_lang_list[$lang_index]}/${decode_set}_hires \
				$dir/${decode_lang_list[$lang_index]}/decode_${decode_set}__tgpr_sri || exit 1;
	  touch $dir/${decode_lang_list[$lang_index]}/decode_${decode_set}__tgpr_sri/.done
	fi
      ) &
    done
  done
fi
