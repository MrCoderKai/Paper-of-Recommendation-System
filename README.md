<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# Paper-of-Recommendation-System
# 协同过滤
1. 《Item-Based Collaborative Filtering Recommendation Algorithms》影响最广的，被引用的次数也最多的一篇推荐系统论文。
文章很长，非常详细地探讨了基于Item-based 方法的协同过滤，作为开山之作，大体内容都是很基础的知识。文章把Item-based算法分为两步：

	* 相似度计算，得到各item之间的相似度

			基于余弦（Cosine-based）的相似度计算
			基于关联（Correlation-based）的相似度计算
			调整的余弦（Adjusted Cosine）相似度计算
	* 预测值计算，对用户未打分的物品进行预测

		加权求和。用户u已打分的物品的分数进行加权求和，权值为各个物品与物品i的相似度，然后对所有物品相似度的和求平均，计算得到用户u对物品i打分。
		回归。如果两个用户都喜欢一样的物品，因为打分习惯不同，他们的欧式距离可能比较远，但他们应该有较高的相似度 。在通过用线性回归的方式重新估算一个新的R(u,N).
$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$