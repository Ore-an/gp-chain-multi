#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal

#
# Copyright 2016 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Bogdan Vlasenko, February 2016
#

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script shows the steps needed to build a recognizer for certain languages
# of the GlobalPhone corpus.
# !!! NOTE: The current recipe assumes that you have pre-built LMs.
echo "This shell script may run as-is on your system, but it is recommended
that you run the commands one by one by copying and pasting into the shell."
#exit 1;
[ -f cmd.sh ] && source ./cmd.sh || echo "cmd.sh not found. Jobs may not execute properly."

# CHECKING FOR AND INSTALLING REQUIRED TOOLS:
#  This recipe requires shorten (3.6.1) and sox (14.3.2).
#  If they are not found, the local/gp_install.sh script will install them.
# local/gp_check_tools.sh $PWD path.sh || exit 1;

. path.sh || { echo "Cannot source path.sh"; exit 1; }
# Set the locations of the GlobalPhone corpus and language model
 
GP_CORPUS=/group/corporapublic/global_phone

GP_LM=/disk/data4/acarmant/gp-chain-multi/data/GPLM

# Set the languages that will actually be processed
export GP_LANGUAGES="WSJ"


# Used to skip parts of the script
stage=4
# The following data preparation step actually converts the audio files from
# shorten to WAV to take out the empty files and those with compression errors.
if [ $stage -le 1 ]; then
  local/gp_data_prep.sh --config-dir=$PWD/conf --corpus-dir=$GP_CORPUS --languages="$GP_LANGUAGES" || exit 1;
  wait;

  local/gp_dict_prep.sh --config-dir $PWD/conf $GP_CORPUS $GP_LANGUAGES || exit 1;
  wait;
fi

if [ $stage -le 2 ]; then
  for L in $GP_LANGUAGES; do
      utils/prepare_lang.sh --position-dependent-phones true \
        data/$L/local/dict "<unk>" data/$L/local/lang_tmp data/$L/lang \
        >& data/$L/prepare_lang.log || exit 1;
  done
  wait;
fi

# Convert the different available language models to FSTs, and create separate
# decoding configurations for each.

if [ $stage -le 3 ]; then
  for L in $GP_LANGUAGES; do
     local/gp_format_lm.sh --filter-vocab-sri true $GP_LM $L &
  done
  exit 0;
  wait;
fi

# Now make MFCC features.
if [ $stage -le 4 ]; then
  for L in $GP_LANGUAGES; do
    mfccdir=mfcc/$L
    for x in train dev eval; do
      (
       steps/make_mfcc.sh --nj 10 --cmd "$train_cmd" data/$L/$x \
          exp/$L/make_mfcc/$x $mfccdir;
       wait;
       steps/compute_cmvn_stats.sh data/$L/$x exp/$L/make_mfcc/$x $mfccdir;
      ) &
    done
  done
  wait;
fi
