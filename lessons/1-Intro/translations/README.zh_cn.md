# 人工智能入门

![本文 人工智能入门 内容总结的涂鸦](../../sketchnotes/ai-intro.png)

>  由 [Tomomi Imura](https://twitter.com/girlie_mac) 绘制

## [课前测验](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/101)

**人工智能** (Artificial Intelligence)是一个令人激动的科学领域，主要研究如何让计算机表现智能行为，比如做那些人类擅长的事。

[Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage)发明了最初的计算机，用来给数字执行明确定义的程序，也就是算法。尽管现代计算机远比19世纪提出的原始模型要先进的多，它们仍然遵循同样的受控计算(controlled computations)的理念。因此，如果我们知道为了实现目标而需要做的确切步骤顺序，就有可能对计算机进行编程，使其做某事。

![一张人类的图片](../images/dsh_age.png)

> 由 [Vickie Soshnikova](http://twitter.com/vickievalerie) 提供的图片

> ✅ 根据照片判定一个人的年龄，是一项无法直接编程的任务，因为我们不清楚我们的大脑具体是怎么根据这个图片给出数字的。

---

然而，有一些任务，我们并不明确知道如何解决。比如，我们莫名学会了如何确定一个人的年龄，因为我们见过许多不同年龄的人的例子，但我们无法明确解释我们是如何做到的，也无法编写计算机程序来完成这个任务。这正是**人工智能** (Artificial Intelligence, 简称AI）所感兴趣的任务类型。

✅想一些AI能帮你完成的任务。考虑金融、医学和艺术领域 - 这些领域如何从人工智能中受益？

## 弱AI vs. 强AI

解决特定问题的系统，例如通过照片确定一个人年龄的系统，可以称为**弱人工智能**(Weak AI)，因为我们正在创建的系统仅能完成一项任务，而不是像人类那样可以解决多个任务。当然，从许多角度来看，开发一个具有普遍智能的计算机系统也是非常有趣的，对学习意识哲学的学生而言也是如此。这样的系统被称为**强人工智能**(Strong AI)或**[人工通用智能](https://en.wikipedia.org/wiki/Artificial_general_intelligence)**(Artificial General Intelligence, AGI)



## 智能的定义和图灵测试

在讨论**[智能](https://en.wikipedia.org/wiki/Intelligence)** (Intelligence)这个词时，其中一个问题是，这个词没有明确的定义。有人认为智能与**抽象思维**或**自我意识**有关，但我们无法准确地定义它。

![一只猫的图片](../images/photo-cat.jpg)

> 由Unsplash的[Amber Kipp](https://unsplash.com/@sadmax)提供的[图片](https://unsplash.com/photos/75715CVEJhI)

要了解词汇「智能」的模糊性，试着回答: "猫是否具有智能？" 不同的人可能对这个问题给出不同的答案，因为没有一种普遍接受的测试来证明这个问题。如果你认为这样的测试存在，试着让你的猫参加智商测试...

请花一分钟时间思考一下你如何定义智能。一只能够解决迷宫问题并获取食物的乌鸦是否具有智能？一个孩子智能吗？

---

当讨论人工通用智能(AGI)时，我们需要一种方法，来证明我们创建了一个真正具有智能的系统。[艾伦·图灵](https://en.wikipedia.org/wiki/Alan_Turing)(Alan Turing)提出了一种被称为**[图灵测试](https://en.wikipedia.org/wiki/Turing_test)**的方式。该测试将人类与给定的系统进行比较。由于任何自动化比较都有可能被计算机系统绕过，评估将由人类进行。因此，如果一个人类无法在基于文本的对话中区分真人和计算机系统，那么该系统被认为是具有智能的。

> 2014年，一款在圣彼得堡开发，名为[Eugene Goostman](https://en.wikipedia.org/wiki/Eugene_Goostman)的聊天机器人，通过一种巧妙的伎俩接近通过图灵测试。它在一开始就宣布自己是一个13岁的乌克兰男孩，这解释了它在文本中的知识缺乏和一些不一致之处。在与评委进行5分钟的对话后，该机器人成功说服了30%的评委认为它是人类 - 一个图灵相信到2000年，机器将能够通过的一个标准。然而，我们应该明白，这并不意味着我们已经创造出一个智能系统，也不意味着计算机系统成功欺骗了人类询问者——系统并没有欺骗人类，而是机器人的创建者们欺骗了评委们！

✅你是否曾经被聊天机器人骗过，误以为你在与一个人类交谈？它是如何让你信服的？

## 实现AI的不同方式

假如我们希望让计算机表现得像是人类，我们就需要以某种方式在计算机中模拟人类的思考模式。因此，我们需要尝试理解是什么使人类具有智能。

> 为了能透过编程赋予计算机智能，我们需要了解自己是如何做出决策的。如果你自我反省一下，就会发现有些过程是下意识发生的。比如，我们可以不假思索地分辨猫和狗，而某些其他的判断可能会用到推理。

下面有两种解决这个问题的可能方法：

 自上而下的方法 (符号推理) (Symbolic Reasoning)               | 自下而上的方法 (神经网络) (Neural Networks)                  
 ------------------------------------------------------------ | ------------------------------------------------------------ 
 自上而下的方法模拟人类的推理。这包括从人类提取知识，并以计算机可以理解的方式表达出来。我们还需要开发一种在计算机内部建模**推理**的方式。 | 由下而上的方法模拟人脑的结构，它由大量的简单单元，**神经元**(neuron)所组成。每个神经元的行为类似于其输入的加权平均，我们可以通过提供**训练数据**来训练神经元网络以解决有用的问题。 

还有一些其他可能实现智能的方法: 

- 复杂的智能行为可以从大量简单单元的互动中产生，**涌现(emergent)**，**协同式(synergetic)**，或**多智能体系统(multi-agent approach)**的设计便是基于此。根据[进化控制论(evolutionary cybernetics)](https://en.wikipedia.org/wiki/Global_brain#Evolutionary_cybernetics)，智能可以在*元系统转变* (metasystem transition)的过程中从更简单、反应性的行为中*出现*。
- 进化法(evolutionary approach)，或是基因算法(genetic algorithm)是基于进化论的优化过程

这些方法会在接下来的课程中讨论，但现在我们将专注于两个主要方向：自上而下和自下而上这两种方法。

### 自上而下的方法 (Top-Down Approach)

在自上而下的方法 (top-down approach)中，我们尝试对我们的推理过程进行建模。由于我们可以在推理过程中追踪自己的思维过程，我们可以尝试体系化这个过程并把它编程到计算机中。这被称为符号推理 (symbolic reasoning)。

人们头脑中往往会使用一些规则来指导他们的决策过程。比如，当医生对病人进行诊断时，医生或许会先了解到病人有发烧，而因此判断病人可能还发炎了。透过给一个特定问题应用大量的规则，医生能得出最后的诊断。

这种方法很大程度依赖于知识表示法(knowledge representation)和推理(reasoning)。从人类专家脑中提取知识应该是最困难的部分了，因为医生不见得总是能确切的知道他们给出的诊断的原因。有时候解决方案只是出现在他们的脑海中，而没有经过明确的思考和推理。有些任务，比如从照片中确定一个人的年龄，根本无法简化为对知识的操作和推理。

### 自下而上的方法 (Bottom-Up Approach)

除此之外，我们也可以尝试对我们大脑中最简单的元素 -- 神经元(neuron) 进行建模。我们可以在计算机中构建一个所谓的**人工神经网络 (artificial neural network)**，然后通过给它提供示例来教它解决问题。这个过程类似于一个新生儿通过观察来学习他或她周围环境的方式。

✅ 研究一下婴儿学习的方式。婴儿大脑的基本元素有哪些？

> | 那机器学习又是什么? |      |
> |--------------|-----------|
> | 人工智能领域中，计算机通过学习一些数据来解决问题的部分被称为**机器学习(Machine Learning)**。在本课程中，我们不会涉及传统的机器学习，欲知详情，可移步[机器学习入门课程 (Machine Learning for Begginers)](http://aka.ms/ml-beginners)。 |   ![机器学习入门课程](../images/ml-for-beginners.png)   |

## AI简史

人工智能领域始于二十世纪中叶。一开始，符号推理 (symbolic reasoning) 十分流行，也带来了许多重要的成功，包括专家系统(expert system) - 一个能在有限的问题范围内扮演专家的计算机程序。然而，人们很快意识到，这种方法难以规模化。从专家那里获取知识，将其表示为计算机可识别的形式，并保持知识库的准确性，事实证明是一项非常复杂且成本高昂的任务，在许多情况下都不切实际。这导致了1970年代所谓的[AI冬季](https://en.wikipedia.org/wiki/AI_winter)的出现。

<img alt="AI简史" src="../images/history-of-ai.png" width="70%"/>

> 由 [Dmitry Soshnikov](http://soshnikov.com) 提供的图片

随着时间的推移，计算资源变得更加便宜，可用的数据越来越多，因此神经网络方法(neural network approaches)开始在许多领域中展示出与人类竞争的出色性能，例如计算机视觉或语音识别。在过去的十年中，人工智能这个术语主要被用作神经网络的代名词，因为我们所听到的大多数人工智能的成功都基于这些方法。



我们可以透过ai国际象棋程序的变化，来观察这些方法的变化

* 早期的国际象棋程序主要基于搜索，会字面意义上预测对手所有可能的动作，并根据几个步骤内可以达到的最佳位置选择最佳走法。这带来了所谓的[alpha-beta剪枝](https://en.wikipedia.org/wiki/Alpha–beta_pruning)(alpha-beta pruning) 搜索算法的发展。
* 搜索策略在游戏的后期表现良好，因为可以落子的地方不多。然而，在游戏初期阶段，搜索空间非常庞大，算法可以通过学习现有的人类对局来改进。后续的实验采用了所谓的[基于案例的推理(case-based reasoning)](https://en.wikipedia.org/wiki/Case-based_reasoning)，程序会在知识库中寻找与当前局面非常相似的案例。
* 现代能击败人类选手的程序基于神经网络(neural networks)和[强化学习(reinforcement learning)](https://en.wikipedia.org/wiki/Reinforcement_learning)，这些程序通过长时间的自我对弈和从自身错误中学习来掌握下棋的技巧，就像人类学习下棋一样。然而，计算机程序可以在短时间内进行大量对局，因此能够更快地学习。

✅ 调查一下AI还玩过哪些游戏

同样地，我们也可以看到实现“聊天程序” (可能通过图灵测试) 的方法转变: 

* 早期的这类程序，如[Eliza](https://en.wikipedia.org/wiki/ELIZA)，主要基于十分简单的语法规则，以及重组输入句子来生成问题的方法。
* 现代的助手程序，如Cortana、Siri或Google Assistant，都是混合系统，使用神经网络将语音转换为文本并识别我们的意图，然后使用一些推理或明确的算法来执行所需的操作。
* 在未来，我们可以期待完全基于神经网络的模型来自主处理对话。最近的GPT和[图灵自然语言生成模型 (Turing-NLG)](https://turing.microsoft.com/)系列的神经网络在这方面取得了巨大的成功。

<img alt="图灵测试的进化" src="../images/turing-test-evol.png" width="70%"/>

> 由Dmitry Soshnikov提供的图片，图中[照片](https://unsplash.com/photos/r8LmVbUKgns) 由Unsplash上的[Marina Abrosimova](https://unsplash.com/@abrosimova_marina_foto)拍摄

## 近期的AI研究

最近的神经网络研究的爆发始于2010年左右，那时大规模的公共数据集开始变得可用。一个包含约1400万个标注图像的，名为ImageNet的大型图像数据集，催生了[ImageNet大规模视觉识别挑战赛 (ImageNet Large Scale Visual Recognition Challenge)](https://image-net.org/challenges/LSVRC/)。

![ILSVRC挑战赛 准确度](../images/ilsvrc.gif)

> 由[Dmitry Soshnikov](http://soshnikov.com) 提供的图片

2012年，[卷积神经网络(Convolutional Neural Networks)](../../4-ComputerVision/07-ConvNets/README.md) 被首次运用于图像分类，大大降低了分类错误率(从近30%降至16.4%)。2015年，微软研究院的ResNet架构达到了[人类准确度水平](https://doi.org/10.1109/ICCV.2015.123)。

自那之后，神经网络在许多任务中都表现出色: 

Since then, Neural Networks demonstrated very successful behaviour in many tasks:

---

Year | Human Parity achieved
-----|--------
2015 | [图像分类](https://doi.org/10.1109/ICCV.2015.123) 
2016 | [对话语音识别](https://arxiv.org/abs/1610.05256) 
2018 | [自动机器翻译](https://arxiv.org/abs/1803.05567) (中文到英文) 
2020 | [图片说明生成](https://arxiv.org/abs/2009.13682) 

在过去几年中，我们目睹了大型语言模型（如BERT和GPT-3）取得了巨大的成功。这主要归功于大量现有的通用文本数据，使我们能够训练模型以捕捉文本的结构和含义，于通用文本集上进行预训练，然后将这些模型专门用于更具体的任务。我们将在本课程的后续内容中学习更多关于自然语言处理（Natural Language Processing）的知识。

## 🚀 小挑战

上网冲个浪，找出你认为人工智能被应用得最有效的领域。是在地图应用中，还是在语音转文本服务中，亦或是在游戏中？研究一下这些系统是如何构建的。”

## [课后测验](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/201)

## 复习 & 自学

阅读[这个课程](https://github.com/microsoft/ML-For-Beginners/tree/main/1-Introduction/2-history-of-ML)来复习AI和ML的历史。从那个课程或本课程顶部的漫画选取一个话题，深入研究并理解该话题演变的文化背景

**作业**: [聊聊游戏](../assignment.md)
