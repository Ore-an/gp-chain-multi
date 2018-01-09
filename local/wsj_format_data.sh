#!/bin/bash

# Copyright 2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
#           2015  Guoguo Chen
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

lang_suffix=

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

. ./path.sh || exit 1;

echo "Preparing train and test data"
srcdir=data/WSJ/local/data
lmdir=data/WSJ/local/nist_lm
tmpdir=data/WSJ/local/lm_tmp
lexicon=data/WSJ/local/lang${lang_suffix}_tmp/lexiconp.txt
mkdir -p $tmpdir

for x in train dev eval; do
  mkdir -p data/WSJ/$x
  cp $srcdir/${x}_wav.scp data/WSJ/$x/wav.scp || exit 1;
  cp $srcdir/$x.txt data/WSJ/$x/text || exit 1;
  cp $srcdir/$x.spk2utt data/WSJ/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk data/WSJ/$x/utt2spk || exit 1;
  utils/filter_scp.pl data/WSJ/$x/spk2utt $srcdir/spk2gender > data/WSJ/$x/spk2gender || exit 1;
done


# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo Preparing language models for test

for lm_suffix in tgpr; do
  test=data/WSJ/lang${lang_suffix}_test_${lm_suffix}

  mkdir -p $test
  cp -r data/WSJ/lang${lang_suffix}/* $test || exit 1;

  gunzip -c $lmdir/lm_${lm_suffix}.arpa.gz | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - $test/G.fst

  utils/validate_lang.pl --skip-determinization-check $test || exit 1;
done

echo "Succeeded in formatting data."
rm -r $tmpdir
