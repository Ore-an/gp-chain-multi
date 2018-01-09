#!/bin/bash -u  

[ -f cmd.sh ] && source ./cmd.sh || echo "cmd.sh not found. Jobs may not execute properly."
. path.sh || { echo "Cannot source path.sh"; exit 1; }

dir=data
check=true # returns total length of datasets
subset=false # subsets the datasets
langs=(PO SP FR GE CZ) 
subfld=lenfiles # folders for filelists
dsets=(train dev eval)
hr=1 # length of subset in hours

mkdir -p $subfld

if [ "$check" = true ]; then
  for lang in $langs; do
    for dset in $dsets; do
      wav-to-duration scp:$dir/$lang/$dset/wav.scp ark,t:$subfld/${lang}-${dset}
      python subset.py check $subfld/${lang}-${dset}
    done
  done
fi

if [ "$subset" = true ]; then
  echo "Subsetting"
  for lang in $langs; do
    for dset in $dsets; do
      wav-to-duration scp:$dir/$lang/$dset/wav.scp ark,t:$subfld/${lang}-${dset}
      python subset.py subset -hr $hr $subfld/${lang}-${dset} $subfld/${lang}-${dset}_${hr}h.list 
      utils/subset_data_dir.sh --utt-list $subfld/${lang}-${dset}_${hr}h.list $dir/$lang/$dset $dir/$lang/${dset}_${hr}h
      utils/validate_data_dir.sh $dir/$lang/${dset}_${hr}h
      wav-to-duration scp:$dir/$lang/${dset}_${hr}h/wav.scp ark,t:$subfld/${lang}-${dset}_${hr}h
      python subset.py check $subfld/${lang}-${dset}_${hr}h
    done
  done
fi
