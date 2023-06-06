#!/usr/bin/env bash
source /e/miniconda/etc/profile.d/conda.sh
# shellcheck disable=SC2215
echo '-----------------------------------------------'
for ((i = 4; i <= 10; i = i + 1)); do
  #  echo 'asd'$i
  trap exit 0 SIGINT
  echo 'start out '$i

  result=$(python main.py --phase train --channel 2 --model efficientnet --ifft True --batch_size 8 --init_lr 5e-5 --num_classes 9 --data_dir openset_annotations/annotations$(($i - 1)))
  echo 'finished out '$i
  echo '--------------------------------------------------'
  mv -f weight_of_model RESULTS/$(($i))_class_out
done

echo '输入任意键结束..'
read zero
exit 0
