"""A tiny Random Forest classification example using scikit-learn."""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    # 1) 造一个二分类的模拟数据集
    features, labels = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=42,
    )

    # 2) 划分训练集/测试集
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # 3) 训练随机森林模型
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42,
    )
    model.fit(x_train, y_train)

    # 4) 预测并评估
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print("Feature importances:", model.feature_importances_)


if __name__ == "__main__":
    main()
