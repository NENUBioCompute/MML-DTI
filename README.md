# MML-DTI

MML-DTI：The MML-DTI framework consists of four components: Input, feature decoder, multi-manifold feature fusion module, and interaction prediction module. For drugs, the framework extracts hyperbolic graph neural network features, molecular fingerprints, and pretrained language model features. These representations are then integrated through the multi-manifold feature fusion module to obtain the final drug feature representation. For targets, the final feature representation is directly derived from pretrained language model features. Finally, the drug and target representations are fed into the interaction prediction module to predict potential interactions.

## MML-DTI

<div align="center">
<p><img src="fig_model.png" width="600" /></p>
</div>

环境
按requirements.txt安装好需要的库
premodel
于https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/tree/main下载molformer，移动到preprocess目录下
于https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main下载ESM2_150M，移动到preprocess目录下
data
将drugbank、Davis、KIBA数据集置放于preprocess下raw_data目录下，下载链接：
Run OurDTI
1.数据处理：执行/preprocess/raw_data文件夹下的txt2csv.py得到各个数据集的csv版本，然后执行/data/split_dataset.py，选择划分random或者冷启动设置
2.执行preprcess文件夹下的get_embeddings.py来获取进入模型的蛋白嵌入与药物嵌入，另外也可以通过执行smilesembedding.py和protembeddings来单独获得药物与蛋白嵌入
3.执行main.py
