# Modified-TOAD-based

## 图像组织区域分割

### 运行命令
python create_patches.py --source  --save_dir  --patch_size --seg --patch --stitch 

### 参数解释
* source：图像保存地址 (eg./DATA_DIRECTORY）
```bash
DATA_DIRECTORY/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```
* save_dir：结果保存地址 (eg./RESULTS_DIRECTORY）

```bash
RESULTS_DIRECTORY/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```
* patch_size：分割大小  (eg.256)
* seg： （不用动）
* patch： （不用动）
* stitch：（不用动）



## 特征提取

### 运行命令
CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_h5_dir  --data_slide_dir  --csv_path  --feat_dir  --resnet_type  --batch_size  --slide_ext 

### 参数解释
* data_h5_dir：分割结果保存地址 (eg./RESULTS_DIRECTORY/)
* data_slide_dir：图像保存地址 (eg./DATA_DIRECTORY)
* csv_path：分割保存的csv文件 (eg./RESULTS_DIRECTORY/process_list_autogen.csv)
* feat_dir：提取的特征保存地址：(eg./FEATURES_DIRECTORY)
* resnet_type:提取特征的模型（默认resnet18) (eg. Resnet50)
* batch_size: 包大小 (eg.512)
* slide_ext：图像格式 (eg. .svs)



## 分割数据集（默认:按照7：2：1方式分割成训练/测试/验证）

### 运行命令
python create_splits.py  --csv_path  --split_dir  --seed   --k 

### 参数解释
* csv_path：保存带有标签的数据列表 (.csv) (eg.SLIDE_PATH/XX.csv)
-----------------------------------------------
患者_id	  | 图像_id  | 标签     |  临床信息
-----------------------------------------------
patient_0 | slide_0	 |subtype_1 |
-----------------------------------------------
patient_1 | slide_1	 |subtype_3 |
-----------------------------------------------
patient_2 | slide_2	 |subtype_2 |
-----------------------------------------------
patient_2 | slide_3	 |subtype_2 |
-----------------------------------------------

* split_dir: 保存分割的地址 (file格式) (eg. SAVE_PATH/)
* seed:随机种子 (eg. 0)
* k:随机分割次数 (eg. 5)



## 训练数据

### 运行命令
CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --early_stopping --lr  --k   --split_dir  --exp_code   --log_data --results_dir  --csv_path  --weighted_sample --model_type  --data_root_dir  

### 参数解释
* drop_out： 是否使用drop层（不用动）
* early_stopping：是否使用早停（不用动）
* lr：学习率 (eg.2e-3)
* k: 训练次数 (eg. 5)
* split_dir：分割保存地址 (eg. SAVE_PATH/)
* exp_code：结果保存的文件夹名 (eg.RESULT_file)
* log_data: 日志（不用动）
* results_dir：结果保存地址 eg. RESULT_PATH/)
* csv_path：保存带有标签的数据列表 (.csv) (eg.SLIDE_PATH/XX.csv)
-----------------------------------------------
患者_id	  | 图像_id  | 标签     |  临床信息
-----------------------------------------------
patient_0 | slide_0	 |subtype_1 |
-----------------------------------------------
patient_1 | slide_1	 |subtype_3 |
-----------------------------------------------
patient_2 | slide_2	 |subtype_2 |
-----------------------------------------------
patient_2 | slide_3	 |subtype_2 |
-----------------------------------------------
* weighted_sample：权重（不用动）
* model_type: 模型类型 （eg. clam_sb）
* data_root_dir：保存提取的图像特征的文件夹 (eg. FEATURE_PATH/H5_PT/)

*** 注意 ***
如果图像和临床共同训练，则需要在./models/model_clam_mcb.py 文件的使用的模块（eg. CLAM_SB)中修改 other_feature参数(默认为TRUE) 为"False"


## 测试数据

### 运行命令
CUDA_VISIBLE_DEVICES=0 python eval.py --csv_path --data_root_dir  --results_dir --save_exp_code  --model_type --drop_out --k  

### 参数解释
* csv_path：保存带有标签的数据列表 (.csv) (eg.SLIDE_PATH/XX.csv)
* data_root_dir：测试保存的图像特征文件夹 (eg. FEATURE_PATH/H5_PT/)
* results_dir：保存模型的位置 (eg. MODEL_RESULTS/)
* save_exp_code: 测试结果保存文件夹 (eg. TEST_RESULTS/)
* model_type: 模型类型 （eg. clam_sb）
* drop_out： 是否使用drop层（不用动）
* k: 训练次数 (eg. 5)


## 绘制热图

###运行命令
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config config_template.yaml

### 注意
文件路径需要在 **config_template.yaml**中修改
