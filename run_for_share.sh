#!/bin/bash

for i in 1 2
do
	for j in 1 2 3
	do
		for lr in 1e-2 1e-3
		do
			for wd in 1e-4 1e-5
			do
				for epoch in 200 300
				do
					for hid in 100 200
					do
    					python main.py --tr 0.6 --path ./datasets/ --dropout 0.1 --weight_decay $wd --alpha 1e-4 --lr $lr --epochs $epoch --dataset BlogCatalog1 --nin $i --nout $j --hidden $hid --clip 100.
					done
				done
			done
		done
	done
done
