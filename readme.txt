-----------------------------------------------------------
show_result.py使用須知

環境要求:
1.python
2.pytroch
3.torchvision
4.numpy
5.PIL
6.tqdm

參數調整:
1.bounding box threshold
2.mask threshold
3.data folder

若要用現有的Sample測試:
創一個資料夾dataset後將雲端上的Sample丟入。

用自己的資料:
創一個資料夾dataset後再裡面創一個資料夾Sample，將照片丟入。

# 04/09/2022更新
補充:不管用雲端上sample的資料或自己的資料都需要有與Sample內相對應的資料夾，才會有照片產出。
若是不想自己創，可以執行一次mkdir.py。


目前測試環境:
1.python3.8
2.pytorch 1.12.1
3.numpy 1.23.1
4.pillow 9.2.0
5.tqdm 4.64.0
6.torchvision 0.13.1
-----------------------------------------------------------
