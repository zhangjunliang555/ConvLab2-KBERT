# ConvLab-2_KBERT

一、目的

1. 利用K-BERT改造convlab2的nlu模块，让模型训练和推理融入知识效果能提升

 

二、结论

1. 知识对少数据有提升

2. 统计上对比情况下，知识几乎无提升，但是体验上有提升

3. 体验上对多意图的top1有提升，top2提升但是导致了噪音

4. 因为训练数据过滤调了无意图语料，导致命名实体数据少了几万条，命名实体的泛化性降低，joinbert联合训练意图和实体对数据要求很高，落地不建议采用

5. 过滤无意图语料并增加知识，比全语料并不增加知识体验效果差，知识起到了噪声作用
6. 开源贡献
   https://gitee.com/zhang-junliang/conv-lab-2_-kbert/tree/master
   
   https://github.com/zhangjunliang555/ConvLab2-KBERT
 

三、背景

1.convlab2

ConvLab是微软美国研究院和清华联合推出了一款开源的多领域端到端对话系统平台，它包括一系列的可复用组件，比如传统的管道系统（pipline systems：包括多个独立步骤的对话系统）或者端对端的神经元模型。方便研究者可以快速使用这些可复用的组件搭建实验模型。同时，ConvLab还提供了一批标注好的数据集和用这些数据集训练好的的预训练模型。

https://github.com/ConvLab/ConvLab

https://github.com/thu-coai/ConvLab-2

 

2. K-BERT

北大开源额将知识融合的输入语料中，即输入的含知识的句子树，利用可见矩阵让知识只对语料中的知识键起作用，对其他不可见，降低知识噪音影响。选择原因，有代码且是语料和知识是中文的。


K-BERT总体架构图

当一个句子“Tim Cook is currently visiting Beijing now”输入时，首先会经过一个知识层（Knowledge Layer），知识层将知识图谱中关联到的三元组信息（Apple-CEO-Tim Cook、Beijing-capital-China 等）注入到句子中，形成一个富有背景知识的句子树（Sentence tree）。

K-BERT 通过软位置编码恢复句子树的顺序信息，即“[CLS](0) Tim(1) Cook(2) CEO(3) Apple(4) is(3) visiting(4) Beijing(5) capital(6) China(7) is_a(6) City(7) now(6)”,可以看到“CEO(3)”和“is(3)”的位置编码都 3，因为它们都是跟在“Cook(2)”之后。

K-BERT 中最大的亮点在于 ***\*Mask-Transformer\****，其中使用了可见矩阵（Visible matrix）将图或树结构中的结构信息引入到模型中。红色表示对应位置的两个 token 相互可见，白色表示相互不可见。

![img](file:///C:\Users\ZHANGJ~1\AppData\Local\Temp\ksohtml\wpsC8B7.tmp.png) 

句子树转化为Embedding representation和可见矩阵

除了软位置和可见矩阵，其余结构均与 Google BERT 保持一致，这就给 K-BERT 带来了一个很好的特性——***\*兼容 BERT 类的模型参数\****。K-BERT 可以直接加载 Google BERT、Baidu ERNIE、Facebook RoBERTa 等市面上公开的已预训练好的 BERT 类模型，无需自行再次预训练，给使用者节约了很大一笔计算资源。

总体而言，知识图谱适合用于提升需要背景知识的任务，而对于不需要背景知识的开放领域任务往往效果不是很显著。

目前 K-BERT 还存在很多问题需要被解决，例如：当知识图谱质量过差时如何提升模型的鲁棒性；在实体关联时如何剔除因一词多义造成的错误关联

 

https://github.com/autoliuweijie/K-BERT
https://zhuanlan.zhihu.com/p/101302240?utm_source=qq

 

3. 其他知识调研

ERNIE，基于掩码，百度版，清华版
K-Adapter，K-BERT，兼容性好
KnowBERT，注意力
CoLAKE，上下文本
KEPLER，WSKLM 多任务

 

https://mp.weixin.qq.com/s/3V6vzrev4xPYKJmGqXNnAg

 

1）ERNIE：
a）百度：
https://github.com/PaddlePaddle/ERNIE
ERNIE 1.0 Base 中文
2.0都是英文

b）清华
https://github.com/thunlp/ERNIE
FewRel 是以 Wikipedia 作为语料库，以 Wikidata 作为知识图谱构建的。
英文数据集

2) K-Adapter
https://github.com/microsoft/K-Adapter
英文
3) K-BERT
https://github.com/autoliuweijie/K-BERT
有中文训练集
4) KnowBERT
https://github.com/allenai/kb
英文

5) CoLAKE
https://github.com/txsun1997/CoLAKE
英文
6) KEPLER
https://github.com/THU-KEG/KEPLER
清华的论文
Wikidata5M dataset 英文
7)WSKLM
无代码实现

8)EmbedKGQA
https://github.com/malllabiisc/EmbedKGQA
https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA

下载不了

 

四、convlab2代码中移植入k-bert代码

1.模型修改

convlab2/nlu/jointBERT/jointKBERT.py

主要增加了kbert类和推理参数等，强制使用可见矩阵

2. 数据预处理修改

convlab2/nlu/jointBERT/dataloaderKBERT.py

主要是增加知识的数据预处理

3. 训练修改

convlab2/nlu/jointBERT/trainKBERT.py

训练调用修改后知识其他模块代码

4. 后处理修改

convlab2/nlu/jointBERT/crosswoz/postprocess.py

增加支持意图top排序，分词字数多余实体标签数容错

5. 验证测试修改

convlab2/nlu/jointBERT/test_kbert.py

测试集推理统计，调用修改后知识其他模块代码

 

6.推理修改

convlab2/nlu/jointBERT/crosswoz/nlu.py

仿照训练和验证修改

 

 

五、测试结果

1. 在用户模型usr数据下，过滤掉无意图数据，增加知识，提高了意图体验

1）无知识

[58000|61410] step

​     slot loss: 0.0007759727934510039

​     intent loss: 0.0015001124116161009

4098 samples val

​     slot loss: 0.032624425354512064

​     intent loss: 0.004649666040766621

--------------------intent--------------------

​     Precision: 96.32

​     Recall: 96.12

​     F1: 96.22

--------------------slot--------------------

​     Precision: 96.89

​     Recall: 93.92

​     F1: 95.39

--------------------overall--------------------

​     Precision: 96.54

​     Recall: 95.26

​     F1: 95.90

best val F1: old_F1=0.9571821684094661,new_F1=0.9589825469360789

i:-1,j:17,intent_logits:-0.40826866030693054

i:-2,j:2,intent_logits:-0.7338897585868835

BERTNLU:天安门在哪,intent:[['Request', '酒店', '地址', '']]

 

i:-1,j:2,intent_logits:3.0711910724639893

i:-2,j:1,intent_logits:-3.9803667068481445

BERTNLU:天安门广场在哪,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']]

 

i:-1,j:17,intent_logits:0.9238511919975281

i:-2,j:2,intent_logits:-1.7600308656692505

BERTNLU:天安门在哪?,intent:[['Request', '酒店', '地址', '']]

 

i:-1,j:17,intent_logits:-0.8324293494224548

i:-2,j:2,intent_logits:-1.3744951486587524

BERTNLU:天安门在哪里,intent:[['Request', '酒店', '地址', '']]

 

i:-1,j:2,intent_logits:-0.07335702329874039

i:-2,j:17,intent_logits:-0.9687855243682861

BERTNLU:天安门在哪里？,intent:[['Request', '景点', '地址', '']]

 

2）有知识

[36000|40940] step

​     slot loss: 0.0035092813823546292

​     intent loss: 0.003655290231850813

4098 samples val

​     slot loss: 0.027416902298986485

​     intent loss: 0.005587439980082997

--------------------intent--------------------

​     Precision: 95.06

​     Recall: 95.02

​     F1: 95.04

--------------------slot--------------------

​     Precision: 96.67

​     Recall: 93.84

​     F1: 95.23

--------------------overall--------------------

​     Precision: 95.67

​     Recall: 94.56

​     F1: 95.12

best val F1: old_F1=0.949436038514443,new_F1=0.9511551155115511

i:-1,j:2,intent_logits:1.2549113035202026

i:-2,j:17,intent_logits:-0.669488787651062

BERTNLU:天安门在哪,intent:[['Request', '景点', '地址', '']]

 

i:-1,j:2,intent_logits:3.303048849105835

i:-2,j:0,intent_logits:-3.0643234252929688

BERTNLU:天安门广场在哪,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']]

 

i:-1,j:2,intent_logits:0.4339403212070465

i:-2,j:17,intent_logits:0.3183131515979767

BERTNLU:天安门在哪?,intent:[['Request', '景点', '地址', ''], ['Request', '酒店', '地址', '']]

 

i:-1,j:2,intent_logits:1.1319308280944824

i:-2,j:17,intent_logits:-0.1406252235174179

BERTNLU:天安门在哪里,intent:[['Request', '景点', '地址', '']]

 

i:-1,j:2,intent_logits:0.9000781178474426

i:-2,j:17,intent_logits:-0.26244470477104187

BERTNLU:天安门在哪里？,intent:[['Request', '景点', '地址', '']]

 

2. 在all模型下（数据增加），过滤掉无意图数据，增加知识，效果不明显，还有相反作用，导致了多标签之间的差距变小

1)无知识

[69000|74025] step

​     slot loss: 0.0008680952598056137

​     intent loss: 0.0018737352865628055

4942 samples val

​     slot loss: 0.04760923078500483

​     intent loss: 0.004257848074226188

--------------------intent--------------------

​     Precision: 95.79

​     Recall: 95.43

​     F1: 95.61

--------------------slot--------------------

​     Precision: 95.29

​     Recall: 92.55

​     F1: 93.90

--------------------overall--------------------

​     Precision: 95.60

​     Recall: 94.30

​     F1: 94.94

best val F1: old_F1=0.9479024756893879,new_F1=0.9494455929336592

 

i:-1,j:2,intent_logits:4.2030558586120605

i:-2,j:17,intent_logits:-0.7924590706825256

BERTNLU:天安门在哪,intent:[['Request', '景点', '地址', '']]

i:-1,j:2,intent_logits:4.810519218444824

i:-2,j:0,intent_logits:-1.3032212257385254

BERTNLU:天安门广场在哪,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']]

 

i:-1,j:2,intent_logits:3.604090929031372

i:-2,j:17,intent_logits:-0.6721656322479248

BERTNLU:天安门在哪?,intent:[['Request', '景点', '地址', '']]

i:-1,j:2,intent_logits:6.5074992179870605

i:-2,j:0,intent_logits:-3.0534961223602295

BERTNLU:天安门广场在哪?,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']]

 

i:-1,j:2,intent_logits:3.5601136684417725

i:-2,j:19,intent_logits:-0.5922917723655701

BERTNLU:天安门在哪里,intent:[['Request', '景点', '地址', '']]

i:-1,j:2,intent_logits:5.966341495513916

i:-2,j:0,intent_logits:-1.2170329093933105

BERTNLU:天安门广场在哪里,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']]

 

i:-1,j:2,intent_logits:2.460899829864502

i:-2,j:19,intent_logits:-0.995185911655426

BERTNLU:天安门在哪里？,intent:[['Request', '景点', '地址', '']]

i:-1,j:2,intent_logits:6.95703649520874

i:-2,j:17,intent_logits:-4.367966175079346

BERTNLU:天安门广场在哪里？,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']

 

2)有知识

[62000|74025] step

​     slot loss: 0.0015058139523395156

​     intent loss: 0.002231120644813927

4942 samples val

​     slot loss: 0.04254680207584923

​     intent loss: 0.0046211492902934895

--------------------intent--------------------

​     Precision: 95.65

​     Recall: 94.92

​     F1: 95.29

--------------------slot--------------------

​     Precision: 95.46

​     Recall: 92.50

​     F1: 93.96

--------------------overall--------------------

​     Precision: 95.58

​     Recall: 93.97

​     F1: 94.77

best val F1: old_F1=0.9450766481707891,new_F1=0.9476755128929043

 

i:-1,j:2,intent_logits:5.445148944854736

i:-2,j:1,intent_logits:2.18560528755188

BERTNLU:天安门在哪,intent:[['Request', '景点', '地址', ''], ['Request', '景点', '名称', '']]

i:-1,j:2,intent_logits:5.054051399230957

i:-2,j:0,intent_logits:-3.5041491985321045

BERTNLU:天安门广场在哪,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']]

 

i:-1,j:2,intent_logits:5.552879810333252

i:-2,j:1,intent_logits:1.7887922525405884

BERTNLU:天安门在哪?,intent:[['Request', '景点', '地址', ''], ['Request', '景点', '名称', '']]

i:-1,j:2,intent_logits:5.7439422607421875

i:-2,j:1,intent_logits:-3.8229117393493652

BERTNLU:天安门广场在哪?,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']]

 

i:-1,j:2,intent_logits:5.482381820678711

i:-2,j:1,intent_logits:0.8291424512863159

BERTNLU:天安门在哪里,intent:[['Request', '景点', '地址', ''], ['Request', '景点', '名称', '']]

i:-1,j:2,intent_logits:5.344071865081787

i:-2,j:1,intent_logits:-4.2407379150390625

BERTNLU:天安门广场在哪里,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']]

 

i:-1,j:2,intent_logits:6.0159382820129395

i:-2,j:1,intent_logits:-0.8315372467041016

BERTNLU:天安门在哪里？,intent:[['Request', '景点', '地址', '']]

i:-1,j:2,intent_logits:6.151431560516357

i:-2,j:1,intent_logits:-5.961650371551514

BERTNLU:天安门广场在哪里？,intent:[['Request', '景点', '地址', ''], ['Inform', '景点', '名称', '天安门广场']]

 

3.训练数据中“天安门广场”要比“天安门”多很多，所以预测“天安门广场”更稳定

 

 

六、遗留问题

1. 为了充分利用命名实体语料，是用用两个joinbert模型还是用单独bert模型,软joinbert

2. 知识利用不高效，推理和训练都要知识查找，知识如何高效应用，

3. 输入预处理统一，增加知识和不增加知识模型不能通用

4. 避免知识噪声

5. 知识与普通语料权重控制

6. 对话的下一步优化方向：软joinbert?知识？多模态？端到端？pipeline其他模块优化？

​	

 



