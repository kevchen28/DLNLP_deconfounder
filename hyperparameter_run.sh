for i in 1 2
do
	for j in 1 2 3
	do
    	python main.py --tr 0.6 --path ./datasets/ --dropout 0.1 --weight_decay 1e-4 --alpha 1e-4 --lr 1e-2 --epochs 200 --dataset BlogCatalog \
	 --norm 1 --nin $i --nout $j --hidden 100 --clip 100.
	done
done