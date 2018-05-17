# mnist_example

## training
```
python train.py -e 100 -g 0 -b 100 -s result
```
```
-e *** : epoch数
-g *** : 使用するgpuカードの番号(cpu実行の場合は、-1にする)
-b *** : ミニバッチのサイズ
-s *** : モデルファイル等を保存するディレクトリ
```
## test
```
python test.py -g 0 -m result/mnist_ex.model -s result
```
```
-g *** : 使用するgpuカードの番号(cpu実行の場合は、-1にする)
-m *** : 使用する学習済みモデル
-s *** : モデルファイル等を保存するディレクトリ
```
