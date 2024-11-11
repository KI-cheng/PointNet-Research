from PointNet import *
from config import DATA_PATH


def predict_single_file(file_path, model_path):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet()

    # 修改加载方式，添加 map_location 和 weights_only=True
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,  # 解决 CUDA/CPU 设备不匹配问题
            weights_only=True  # 提高安全性
        )
    )

    model.to(device)
    model.eval()

    # 数据预处理
    transforms = Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])

    # 读取和处理文件
    with open(file_path, 'r') as f:
        mesh = read_off(f)

    # 获取处理后的点云数据
    processed_pointcloud = transforms(mesh)  # 这里会得到处理后的点云数据

    pointcloud = processed_pointcloud.unsqueeze(0)
    pointcloud = pointcloud.to(device).float()

    # 预测
    with torch.no_grad():
        outputs, _, _ = model(pointcloud.transpose(1, 2))
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = predicted.item()

    # 返回预测类别和处理后的点云数据
    return {
        'predicted_class': CLASSES[predicted_class],
        'processed_points': processed_pointcloud.cpu().numpy().tolist()  # 转换为Python列表
    }


if __name__ == "__main__":
    model_path = './PTH/best_pointnet10_cls.pth'
    file_path = f'{DATA_PATH}/toilet/train/toilet_0001.off'  # 替换为实际的OFF文件路径

    predicted_class = predict_single_file(file_path, model_path)
    print(predicted_class['predicted_class'], len(predicted_class['processed_points']))
