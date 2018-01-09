#!/bin/bash -u
[ -f cmd.sh ] && source ./cmd.sh || echo "cmd.sh not found. Jobs may not execute properly."

. path.sh || { echo "Cannot source path.sh"; exit 1; }

export GP_LANGUAGES="PO"

# Used to skip parts of the script
stage=100
trn_only=-1
dec_only=1

# Train monophone model
if [ $stage -le 1 ] || [ $trn_only -ge 1 ]; then
  for L in $GP_LANGUAGES; do
    mkdir -p exp/$L/mono;
    steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
      data/$L/train data/$L/lang exp/$L/mono >& exp/$L/mono/train.log &
  done
  wait;
fi


# Decode monophone model
if [ $stage -le 2 ] || [ $dec_only -ge 1 ]; then
  for L in $GP_LANGUAGES; do
    for lm_suffix in tgpr_sri; do
      (
        graph_dir=exp/$L/mono/graph_${lm_suffix}
        mkdir -p $graph_dir
	utils/mkgraph.sh data/$L/lang_test_${lm_suffix} exp/$L/mono \
  		 $graph_dir
	wait;
	steps/decode.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/dev \
  			exp/$L/mono/decode_dev_${lm_suffix}
	steps/decode.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/eval \
  			exp/$L/mono/decode_eval_${lm_suffix}
      ) &
    done
  done
  wait;
fi





# Train tri1, which is first triphone pass

if [ $stage -le 3 ] || [ $trn_only -ge 1 ]; then
  for L in $GP_LANGUAGES; do
      mkdir -p exp/$L/mono_ali
      steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  		      data/$L/train data/$L/lang exp/$L/mono \
  		      exp/$L/mono_ali >& exp/$L/mono_ali/align.log
      wait;
      num_states=$(grep "^$L" conf/tri.conf | cut -f2)
      num_gauss=$(grep "^$L" conf/tri.conf | cut -f3)
      mkdir -p exp/$L/tri1
      steps/train_deltas.sh --cmd "$train_cmd" \
       --cluster-thresh 100 $num_states $num_gauss data/$L/train data/$L/lang \
       exp/$L/mono_ali exp/$L/tri1 >& exp/$L/tri1/train.log
  done
  wait;
fi


# Decode tri1
if [ $stage -le 4 ] || [ $dec_only -ge 1 ]; then
  for L in $GP_LANGUAGES; do
    for lm_suffix in tgpr_sri; do
      (
        graph_dir=exp/$L/tri1/graph_${lm_suffix}
        mkdir -p $graph_dir
        utils/mkgraph.sh data/$L/lang_test_${lm_suffix} exp/$L/tri1 \
  	$graph_dir
        wait;
        steps/decode.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/dev \
  	exp/$L/tri1/decode_dev_${lm_suffix}
        steps/decode.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/eval \
  	exp/$L/tri1/decode_eval_${lm_suffix}
      ) &
    done
  done
fi




# Train tri2, which is LDA+MLLT
if [ $stage -le 5 ] || [ $trn_only -ge 1 ]; then
  for L in $GP_LANGUAGES; do
      mkdir -p exp/$L/tri1_ali
      steps/align_si.sh --nj 10 --cmd "$train_cmd" data/$L/train \
  	     data/$L/lang exp/$L/tri1 exp/$L/tri1_ali >& exp/$L/tri1_ali/tri1_ali.log
      wait;
      num_states=$(grep "^$L" conf/tri.conf | cut -f2)
      num_gauss=$(grep "^$L" conf/tri.conf | cut -f3)
      mkdir -p exp/$L/tri2
      steps/train_lda_mllt.sh --cmd "$train_cmd" \
  			    --splice-opts "--left-context=3 --right-context=3" \
  			    $num_states $num_gauss data/$L/train data/$L/lang \
  			    exp/$L/tri1_ali \
  			    exp/$L/tri2 >& exp/$L/tri2/tri2.log
  done
  wait;
fi

# Decode tri2
if [ $stage -le 6 ] || [ $dec_only -ge 1 ]; then
  for L in $GP_LANGUAGES; do
    for lm_suffix in tgpr_sri; do
    (
      graph_dir=exp/$L/tri2/graph_${lm_suffix}
      mkdir -p $graph_dir

      utils/mkgraph.sh data/$L/lang_test_${lm_suffix} exp/$L/tri2 $graph_dir
      wait;
      steps/decode.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/dev \
  	   exp/$L/tri2/decode_dev_${lm_suffix}
      steps/decode.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/eval exp/$L/tri2/decode_eval_${lm_suffix}
    ) &
    done
  done
  wait;
fi





# Train tri3, which is LDA+MLLT+SAT
if [ $stage -le 7 ] || [ $trn_only -ge 1 ]; then
  for L in $GP_LANGUAGES; do
       mkdir -p exp/$L/tri2_ali
       steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" --use-graphs true \
			    data/$L/train data/$L/lang exp/$L/tri2 exp/$L/tri2_ali >& exp/$L/tri2_ali/align.log
       wait;
       
       num_states=$(grep "^$L" conf/tri.conf | cut -f2)
       num_gauss=$(grep "^$L" conf/tri.conf | cut -f3)
       
       mkdir -p exp/$L/tri3
       steps/train_sat.sh --cmd "$train_cmd" \
  			       $num_states $num_gauss data/$L/train data/$L/lang \
  			       exp/$L/tri2_ali \
  			       exp/$L/tri3 >& exp/$L/tri3/tri3.log
  done
  wait;
fi

# Decode tri3
if [ $stage -le 8 ] || [ $dec_only -ge 1 ]; then
  for L in $GP_LANGUAGES; do
    for lm_suffix in tgpr_sri; do
    (
      graph_dir=exp/$L/tri3/graph_${lm_suffix}
      mkdir -p $graph_dir
      
      utils/mkgraph.sh data/$L/lang_test_${lm_suffix} exp/$L/tri3 $graph_dir
      wait;
      steps/decode_fmllr.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/dev \
  			    exp/$L/tri3/decode_dev_${lm_suffix}
      steps/decode_fmllr.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/eval \
			    exp/$L/tri3/decode_eval_${lm_suffix}
    ) &
    done
  done
  wait;
fi

# Do alignments for next step (LF-MMI)

if [ $stage -le 9 ] || [ $trn_only -ge 1 ]; then
  for L in $GP_LANGUAGES; do
    mkdir -p exp/$L/tri3_ali
    steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" --use-graphs true \
			 data/$L/train data/$L/lang exp/$L/tri3 \
			 exp/$L/tri3_ali >& exp/$L/tri3_ali/align.log
    wait;
  done
  wait;
fi



# # # Train nnet, which is HMM-NN.
# for L in $GP_LANGUAGES; do
#   (
#       mkdir -p exp/$L/tri2_ali
#       $cpu_cmd steps/align_si.sh --nj 10 --cmd "$train_cmd" --use-graphs true data/$L/train data/$L/lang exp/$L/tri2 exp/$L/tri2_ali >& exp/$L/tri2_ali/align.log

#     num_layers=$(grep "^$L" conf/nn.conf | cut -f2)
#     num_dim=$(grep "^$L" conf/nn.conf | cut -f3)
#     splice=$(grep "^$L" conf/nn.conf | cut -f4)
#     lr=$(grep "^$L" conf/nn.conf | cut -f5)
#     mkdir -p exp/$L/nnet
#     trdir=data/$L/train
#     # utils/subset_data_dir_tr_cv.sh $trdir ${trdir}_tr90 ${trdir}_cv10
#     $nn_cmd steps/nnet/train.sh --hid-layers $num_layers --hid-dim $num_dim --splice $splice --learn-rate $lr \
# 	data/$L/train_tr90 data/$L/train_cv10 data/$L/lang exp/$L/tri2_ali exp/$L/tri2_ali exp/$L/nnet >& exp/$L/nnet/train.log
#   ) &
# done
# wait;

# # Decode nnet
# for L in $GP_LANGUAGES; do
#   for lm_suffix in tgpr_sri; do
#   (
#     graph_dir=exp/$L/nnet/graph_${lm_suffix}
#     mkdir -p $graph_dir
#     $cpu_cmd utils/mkgraph.sh data/$L/lang_test_${lm_suffix} exp/$L/nnet \
# 	$graph_dir

#     mkdir -p exp/$L/nnet/decode_dev_${lm_suffix}
#     $cpu_cmd steps/nnet/decode.sh $graph_dir data/$L/dev exp/$L/nnet/decode_dev_${lm_suffix}
#     mkdir -p exp/$L/nnet/decode_eval_${lm_suffix}
#     $cpu_cmd steps/nnet/decode.sh $graph_dir data/$L/eval exp/$L/nnet/decode_eval_${lm_suffix}
#   ) &
# done
# done
# wait;
