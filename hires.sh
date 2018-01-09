for lang in WSJ; do
  for dir in train dev eval; do
    utils/copy_data_dir.sh data/$lang/${dir} data/$lang/${dir}_hires
    steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf --cmd run.pl  data/$lang/${dir}_hires exp/make_hires/$lang/${dir} mfcc/$lang || exit 1;
  done
done
