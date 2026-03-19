import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main():
    # 1. 配置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B/32" # 你可以根据你的 FedProCLIP 项目更改为对应的模型结构
    save_path = "clip_anchors_tsne.png"

    # 2. 定义类别和提示词模板 (Prompt Templates)
    # 我们为每个类别设置多个锚点模板，以便在 t-SNE 中形成聚类簇
    classes = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]
    templates = [
        "a photo of a {}.",
            "a bad photo of a {}.",
            "a close-up photo of a {}.",
            "a sketch of a {}.",
            "a painting of a {}.",
            "a cropped photo of a {}.",
            "a black and white photo of a {}."
     
    ]

    print(f"Loading CLIP model ({model_name})...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    # 3. 生成锚点文本并提取特征
    all_features = []
    all_labels = []

    print("Extracting text features...")
    with torch.no_grad():
        for class_idx, class_name in enumerate(classes):
            # 组合模板和类别名生成文本锚点
            prompts = [template.format(class_name) for template in templates]
            text_tokens = clip.tokenize(prompts).to(device)
            
            # 获取文本编码器特征
            text_features = model.encode_text(text_tokens)
            
            # L2 归一化 (CLIP 特征通常需要归一化计算相似度)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            fused_feature,_ = text_features.max(dim=0, keepdim=True)
            
            all_features.append(fused_feature.cpu().numpy())
            all_labels.append(class_name) # 每个类别只对应一个标签

    # 合并所有特征 [num_classes * num_templates, feature_dim]
    features_np = np.vstack(all_features)
    labels_np = np.array(all_labels)

    # 4. 使用 t-SNE 进行降维
    print("Running t-SNE...")
    # 注意: perplexity 的值必须小于样本总数。这里样本数较少，所以设置一个较小的 perplexity
    tsne = TSNE(n_components=2, perplexity=3, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features_np)

    # 5. 绘制 t-SNE 图
    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    
    # 不同的类别使用不同的颜色
    colors = plt.cm.get_cmap('tab10', len(classes))

    for i, class_name in enumerate(classes):
        # 提取当前类的 2D 坐标
        point = features_2d[i]
        # 绘制散点
        plt.scatter(
            point[0],
            point[1],
            label=class_name,
            color=colors(i),
            s=100,           # 点的大小
            alpha=0.8,       # 透明度
            edgecolors='w'   # 边缘颜色
        )
        
        # 可选：在每个类别的中心位置添加文本标签
        # center_x, center_y = class_points.mean(axis=0)
        plt.text(point[0] + 2, point[1] + 2, class_name, fontsize=12, fontweight='bold')

    plt.title("t-SNE Visualization of CLIP Text Anchors", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 6. 保存并显示图像
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE plot saved successfully to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()