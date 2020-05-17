# CRNN-ATTENTION
crnn feature net + attention 机制

基于 tensorflow eager 格式编写。

数据集使用
mjsynth.tar.gz    http://www.robots.ox.ac.uk/~vgg/data/text/
需要将train_net.py 中 root 目录指向 mjsynth 中包含 annotation_train.txt 的文件夹

注意:tensorflow eager 保存好像是直接序列化层对象,导致GPU上训练的模型只能在GPU上使用,CPU同理

添加attention机制后,模型收敛很快,基本一个epoch就能基本收敛,所以可以考虑直接在CPU上训练测试

目前版本只能用于检测单个单词,对多个单词的情况基本预测全错
参考attention_ocr看有什么需要改进的地方,可以将ctc loss和attention loss 进行加权将两个模型融合在一起，然后看一下效果
（腾讯精准推荐：我们在卷积层采取类似VGG网络的结构，减少CNN卷积核数量的同时增加卷积层深度，既保证精度，又降低时耗。
在RNN一侧，我们针对LSTM有对语料和图像背景过拟合的倾向，在双向LSTM单元层实现了Dropout策略。
在训练技巧一侧，我们针对CTC loss对初始化敏感和收敛速度慢的问题，采用样本由易到难、分阶段训练的策略。
在测试阶段，针对字符拉伸导致识别率降低的问题，我们保持输入图像尺寸比例，根据卷积特征图的尺寸动态决定LSTM时序长度。
我们使用的算法的网络结构如图5所示，由于以上所述的多处改进，我们的算法速度快且精度高，
在身份证、银行卡等业务上取得98%以上识别准确率。）


