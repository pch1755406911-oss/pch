# pch

一个最小可运行的随机森林示例（Python + scikit-learn）。

## 运行方式

```bash
python random_forest_demo.py
```

如果你还没安装依赖：

```bash
pip install scikit-learn
```

## 代码说明

`random_forest_demo.py` 做了 4 件事：

1. 生成二分类模拟数据。
2. 划分训练集与测试集。
3. 训练 `RandomForestClassifier`。
4. 输出准确率和特征重要性。
