#!/bin/bash
stage=0
train_stage=0
get_egs_stage=-10
speed_perturb=false # not implemented
dir=exp/chain/multiwsj
alidir=tri3_ali
# directory of the experiment the alignments are based on
alibexpdir=tri3

# lang_list is space-separated language list used for multilingual training
lang_list=(PO SP WSJ)           # (FR SP GE PO CZ)
# lang2weight is comma-separated list of weights, one per language, used to
# scale example's output w.r.t its input language during training.
lang2weight=""
# The language list used for decoding.
decode_lang_list=(PO SP WSJ)             # (FR SP GE PO CZ)

# Number of leaves in the new topology tree
leaves=9000

frame_subsampling_factor=3

# TDNN options

dim=625
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=1
num_jobs_final=1
minibatch_size=128
frames_per_eg=150
remove_egs=false
common_egs_dir=
xent_regularize=0.1

echo "$0 $@" # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

set -e
if ! cuda-compiled; then
  cat <<EOF #&& exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

num_langs=${#lang_list[@]}

for lang_index in `seq 0 $[$num_langs-1]`; do
  for f in data/${lang_list[$lang_index]}/train/{feats.scp,text} \
		exp/${lang_list[$lang_index]}/$alidir/ali.1.gz; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
done

suffix=""
if [ "$speed_perturb" == "true" ]; then suffix=_sp; fi
## ./utils/data/perturb_data_dir_speed_3way.sh data/$lang/${datadir} data/$lang/${datadir}_sp

dir=${dir}${suffix}


for lang_index in `seq 0 $[$num_langs-1]`; do
  multi_fst_dirs[$lang_index]=$dir/${lang_list[$lang_index]}
  multi_train_dirs[$lang_index]=data/${lang_list[$lang_index]}/train${suffix}
  multi_lang_dirs[$lang_index]=data/${lang_list[$lang_index]}/lang
  multi_ali_dirs[$lang_index]=exp/${lang_list[$lang_index]}/${alidir}${suffix}
  multi_egs_dirs[$lang_index]=exp/chain/multi-${lang_list[$lang_index]}-egs
  multi_tree_dirs[$lang_index]=exp/chain/multi-${lang_list[$lang_index]}-tree
  multi_bexp_dirs[$lang_index]=exp/${lang_list[$lang_index]}/${alibexpdir}
done

if [ $stage -le 1 ]; then
  for l in `seq 0 $[$num_langs-1]`; do
    # Get the alignments as lattices (gives the LF-MMI training more freedom).
    # use the same num-jobs as the alignments
    nj=$(cat ${multi_ali_dirs[$l]}/num_jobs) || exit 1;
    steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${multi_train_dirs[$l]} \
			      ${multi_lang_dirs[$l]} ${multi_bexp_dirs[$l]} \
			      ${multi_bexp_dirs[$l]}_lats$suffix
    rm ${multi_bexp_dirs[$l]}_lats$suffix/fsts.*.gz # save space
  done
fi

# Make trees5A5A

if [ $stage -le 2 ]; then
  for l in `seq 0 $[$num_langs-1]`; do
    # Create a version of the lang/ directory that has one state per phone in the
    # topo file.
    cp -r ${multi_lang_dirs[$l]} ${multi_lang_dirs[$l]}_chain
    silphonelist=$(cat ${multi_lang_dirs[$l]}_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat ${multi_lang_dirs[$l]}_chain/phones/nonsilence.csl) || exit 1;
    # As of now, gen_topo.py puts sil and nonsil together, no difference between them
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist > \
				  ${multi_lang_dirs[$l]}_chain/topo
  done
fi



for l in `seq 0 $[$num_langs-1]`; do
  multi_lats_dirs[$l]=${multi_bexp_dirs[$l]}_lats$suffix
  # Use the newly made topology
  ntopo_multi_lang_dirs[$l]=${multi_lang_dirs[$l]}_chain
done


if [ $stage -le 3 ]; then
  for l in `seq 0 $[$num_langs-1]`; do
    # Build the tree with the new topology
    steps/nnet3/chain/build_tree.sh --cmd "$train_cmd" --leftmost-questions-truncate -1 \
				    --frame-subsampling-factor $frame_subsampling_factor $leaves \
				    ${multi_train_dirs[$l]} ${ntopo_multi_lang_dirs[$l]} \
				    ${multi_ali_dirs[$l]} ${multi_tree_dirs[$l]}
  done
fi

if  [ $stage -le 4 ]; then
    echo "$0: creating phone language-model"
    lm_opts='--num-extra-lm-states=2000'
    for l in `seq 0 $[$num_langs-1]`; do
	mkdir -p ${multi_fst_dirs[$l]}
	cp ${multi_tree_dirs[$l]}/tree ${multi_fst_dirs[$l]}
	$train_cmd ${multi_tree_dirs[$l]}/log/make_phone_lm.log \
		   chain-est-phone-lm $lm_opts \
		   "ark:gunzip -c  ${multi_tree_dirs[$l]}/ali.*.gz | ali-to-phones  ${multi_tree_dirs[$l]}/final.mdl ark:- ark:- |" \
		   ${multi_fst_dirs[$l]}/phone_lm.fst || exit 1;
    done
fi

if [ $stage -le 5 ]; then
    echo "$0: creating denominator FST" 
for l in `seq 0 $[$num_langs-1]`; do
	copy-transition-model  ${multi_tree_dirs[$l]}/final.mdl ${multi_fst_dirs[$l]}/0.trans_mdl
	$train_cmd  ${multi_fst_dirs[$l]}/log/make_den_fst.log \
			      chain-make-den-fst ${multi_fst_dirs[$l]}/tree \
			      ${multi_fst_dirs[$l]}/0.trans_mdl \
			      ${multi_fst_dirs[$l]}/phone_lm.fst \
			      ${multi_fst_dirs[$l]}/den.fst \
			      ${multi_fst_dirs[$l]}/normalization.fst || exit 1;
done   
fi



if [ $stage -le 6 ]; then
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=$dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$dim
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$dim 
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$dim
  relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=$dim
EOF

  # fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat

  for l in `seq 0 $[$num_langs-1]`; do
    num_targets=`am-info ${multi_fst_dirs[$l]}/0.trans_mdl 2>/dev/null | grep -w pdfs | awk '{print $NF}'` || exit 1;
    
    # out for chain branch
    echo " relu-batchnorm-layer name=prefinal-chain-${l} input=tdnn6 dim=$dim target-rms=0.5"
    echo " output-layer name=output-${l} include-log-softmax=false dim=$num_targets max-change=1.5"
    

    # out for xent regularizing branch
    echo " relu-batchnorm-layer name=prefinal-xent-${l} input=tdnn6 dim=625 target-rms=0.5"
    echo " output-layer name=output-${l}-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5"
  done >> $dir/configs/network.xconfig

  cp  $dir/configs/network.xconfig  $dir/configs/network.xconfig.bk  

  sed -e 's/output-0/output/g' -e 's/output-0-xent/output-xent/g'  $dir/configs/network.xconfig.bk >  $dir/configs/network.xconfig
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
				    --config-dir $dir/configs

  # Don't think it's needed
  #nnet3-copy --edits="rename-node old-name=output new-name=output-0" $dir/configs/init.raw $dir/configs/init.raw
fi


for l in `seq 0 $[$num_langs-1]`;do
    multi_train_dirs[$l]=data/${lang_list[$l]}/train_hires${suffix}
done


if [ $stage -le 7 ]; then
  # generate egs dir
  
  . $dir/configs/vars || exit 1;
  
  local/nnet3/prepare_multilingual_egs_chain.sh --cmd "$decode_cmd" \
				--left-context $[$model_left_context+$frame_subsampling_factor/2] \
				--right-context $[$model_right_context+$frame_subsampling_factor/2] \
			        --cmvn-opts "--norm-means=false --norm-vars=false" \
				--right-tolerance 10 \
				--left-tolerance 10 \
				--frame-subsampling-factor $frame_subsampling_factor \
				--frames-per-eg $frames_per_eg \
				$num_langs ${multi_train_dirs[@]} ${multi_fst_dirs[@]} \
				${multi_lats_dirs[@]} \
				${multi_egs_dirs[@]} || exit 1
  #${multi_ali_dirs[@]}_chain \
fi





    
# if [ $stage -le 9 ]; then
#   # Change so you can just pass in the lists
#   local/nnet3/chain/train_multi.sh --cmd "$decode_cmd" --num-jobs-nnet "1 1" \
# 				   ${multi_lats_dirs[0]} ${multi_tree_dirs[0]} ${multi_egs_dirs[0]} \
# 				   ${multi_lats_dirs[1]} ${multi_tree_dirs[1]} ${multi_egs_dirs[1]} \
# 				   ./exp/chain/PO/0.mdl $dir  || exit 1;
# fi


if [ -z $megs_dir ];then
  megs_dir=$dir/egs
fi

if [ $stage -le 8 ] && [ ! -z $megs_dir ]; then
  # generate separate multilingual egs dir
  if [ ! -z "$lang2weight" ]; then
    egs_opts="--lang2weight '$lang2weight'"
  fi
  common_egs_dir="${multi_egs_dirs[@]} $megs_dir"
  steps/nnet3/multilingual/combine_egs.sh $egs_opts \
					  --cmd "$decode_cmd" \
					  --egs-prefix "cegs." \
					  --minibatch-size $minibatch_size \
					  --samples-per-iter 40000 \
					  $num_langs ${common_egs_dir[@]} || exit 1;
fi


# Takes time, commented to skip
if [ $stage -le 9 ]; then
  echo "Calculating LDA"
  #max_lda_jobs=`exec ls $megs_dir | sed 's/.*\([0-9]\+\).*/\1/g' | sort -n | tail -1`
  #$train_cmd JOB=1:$max_lda_jobs $dir/log/acc_lda_stats.JOB.log \
#	     nnet3-chain-acc-lda-stats --rand-prune=4.0 $dir/configs/init.raw "scp:$megs_dir/cegs.JOB.scp" $dir/JOB.lda_stats
  
 # sum-lda-accs $dir/lda_stats $dir/*.lda_stats
 
  #nnet-get-feature-transform $lda_opts $dir/configs/lda.mat $dir/lda_stats && rm $dir/lda_stats;
  
  nnet3-init $dir/configs/final.config - | nnet3-copy --edits="rename-node old-name=output new-name=output-0;rename-node old-name=output-xent new-name=output-0-xent" - $dir/0.raw
fi

for l in `seq 0 $[$num_langs-1]`;do
  den_fst_to_output="${den_fst_to_output} ${lang_list[$l]}/den.fst:output-${l}"
done

if [ $stage -le 10 ]; then
  
  steps/nnet3/chain/train.py --stage $train_stage \
			     --cmd "$decode_cmd" \
			     --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
			     --chain.xent-regularize $xent_regularize \
			     --chain.leaky-hmm-coefficient 0.1 \
			     --chain.l2-regularize 0.00005 \
			     --chain.apply-deriv-weights false \
			     --chain.lm-opts="--num-extra-lm-states=2000" \
			     --chain.den-fst-to-output="$den_fst_to_output" \
			     --egs.dir $megs_dir \
			     --egs.stage $get_egs_stage \
			     --egs.opts "--frames-overlap-per-eg 0" \
			     --egs.chunk-width $frames_per_eg \
			     --trainer.input-model $dir/0.raw \
			     --trainer.num-chunk-per-minibatch $minibatch_size \
			     --trainer.frames-per-iter 1500000 \
			     --trainer.num-epochs $num_epochs \
			     --trainer.optimization.num-jobs-initial $num_jobs_initial \
			     --trainer.optimization.num-jobs-final $num_jobs_final \
			     --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
			     --trainer.optimization.final-effective-lrate $final_effective_lrate \
			     --trainer.max-param-change $max_param_change \
			     --cleanup.remove-egs $remove_egs \
			     --feat-dir ${multi_train_dirs[0]} \
			     --tree-dir ${multi_tree_dirs[0]} \
			     --lat-dir ${multi_bexp_dirs[0]}_lats$suffix \
			     --dir $dir  || exit 1;
fi


if [ $stage -le 11 ]; then
  for lang_index in `seq 0 $[$num_langs-1]`; do
    lang_dir=$dir/${lang_list[$lang_index]}
    mkdir -p  $lang_dir
    # rename output for each lang, add transition model
    nnet3-copy --edits="rename-node old-name=output-$lang_index new-name=output; rename-node old-name=output-${lang_index}-xent new-name=outputxent" \
	  $dir/final.raw - | nnet3-copy --edits="remove-output-nodes name=output-*; remove-orphans" \
	  - - | nnet3-copy --edits="rename-node old-name=outputxent new-name=output-xent" \
	  - - |  nnet3-am-init ${multi_tree_dirs[$lang_index]}/final.mdl - \
	  $lang_dir/final.mdl || exit 1;
    cp $dir/cmvn_opts $lang_dir/cmvn_opts || exit 1;
    cp ${multi_tree_dirs[$lang_index]}/tree $lang_dir || exit 1;
    #TODO? removing unused layer
  done
fi

num_decode_lang=${#decode_lang_list[@]}

if [ $stage -le 12 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  for lang_index in `seq 0 $[$num_decode_lang-1]`; do
   ( utils/mkgraph.sh --self-loop-scale 1.0 data/${decode_lang_list[$lang_index]}/lang_test_tgpr_sri $dir/${decode_lang_list[$lang_index]} $dir/${decode_lang_list[$lang_index]}/graph_tgpr_sri ) &
   wait;
  done
fi

decode_suff=_tgpr_sri

if [ $stage -le 13 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for lang_index in `seq 0 $[$num_decode_lang-1]`; do
    graph_dir=$dir/${decode_lang_list[$lang_index]}/graph_tgpr_sri
    for decode_set in dev eval; do
      (
	steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --iter final \
			      --nj 5 --cmd "$decode_cmd" $iter_opts \
			      $graph_dir data/${decode_lang_list[$lang_index]}/${decode_set}_hires $dir/${decode_lang_list[$lang_index]}/decode_${decode_set}${decode_iter:+$decode_iter}_${decode_suff} || exit 1;
      ) &
    done
  done
fi

wait;

# if [ $stage -le 14 ]; then
#   for lang_index in `seq 0 $[$num_decode_lang-1]`; do
#     graph_dir=$dir/${decode_lang_list[$lang_index]}/graph_tgpr_sri
#     for decode_set in dev eval; do
#       (
# 	steps/score_kaldi.sh --cmd "$decode_cmd" data/${decode_lang_list[$lang_index]}/${decode_set}_hires \
# 			     $graph_dir $dir/${decode_lang_list[$lang_index]}/decode_${decode_set}${decode_iter:+$decode_iter}_${decode_suff}
#       ) &
#     done
#   done
# fi

wait;
exit 0;


#TODO solidify the folders inside the $dir (single language subfolder for tree, fsts and final model)
