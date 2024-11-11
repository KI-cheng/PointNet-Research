from PointNet import *
from config import DATA_PATH


# 新增的数据增强方法
def generate_sphere(center, radius, num_points):
    u = np.random.uniform(0, 1, num_points)
    v = np.random.uniform(0, 1, num_points)

    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    return np.column_stack((x, y, z))


def apply_occlusion(pointcloud, sphere_center, sphere_radius):
    distances = np.linalg.norm(pointcloud - sphere_center, axis=1)
    occluded_pointcloud = pointcloud[distances > sphere_radius]
    return occluded_pointcloud


def pad_or_crop(pointcloud, target_size):
    current_size = pointcloud.shape[0]
    if current_size < target_size:
        padding = np.zeros((target_size - current_size, pointcloud.shape[1]))
        return np.vstack((pointcloud, padding))
    elif current_size > target_size:
        return pointcloud[:target_size]
    return pointcloud


# def pad_or_crop(pointcloud, target_size):
#     current_size = pointcloud.shape[0]
#     if current_size < target_size:
#         # 从现有点中随机选择点进行填充，而不是用零填充
#         padding_indices = np.random.choice(current_size, target_size - current_size, replace=True)
#         padding = pointcloud[padding_indices]
#         return np.vstack((pointcloud, padding))
#     elif current_size > target_size:
#         # 随机选择点而不是简单截取前N个
#         indices = np.random.choice(current_size, target_size, replace=False)
#         return pointcloud[indices]
#     return pointcloud

class Sphere_Occlusion(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        random_index = np.random.randint(0, pointcloud.shape[0])
        sphere_center = pointcloud[random_index]

        sphere_radius = 0.2

        occluded_pointcloud = apply_occlusion(pointcloud, sphere_center, sphere_radius)
        return pad_or_crop(occluded_pointcloud, 1024)


class RandomShift(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        shift_range = 0.1
        shifts = np.random.uniform(-shift_range, shift_range, pointcloud.shape)
        shifted_pointcloud = pointcloud + shifts
        return pad_or_crop(shifted_pointcloud, 1024)


class RandomScale(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        scale_low = 0.8
        scale_high = 1.25
        scales = np.random.uniform(scale_low, scale_high, (pointcloud.shape[0], 1))
        scaled_pointcloud = pointcloud * scales
        return pad_or_crop(scaled_pointcloud, 1024)


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

    # 数据预处理（包含新的数据增强）
    transforms = Compose([
        PointSampler(1024),
        Normalize(),
        Sphere_Occlusion(),
        RandomShift(),
        RandomScale(),
        ToTensor()
    ])

    # 读取和处理文件
    with open(file_path, 'r') as f:
        mesh = read_off(f)

    processed_pointcloud = transforms(mesh)  # 这里会得到处理后的点云数据

    pointcloud = processed_pointcloud.unsqueeze(0)
    pointcloud = pointcloud.to(device).float()

    # 预测
    with torch.no_grad():
        outputs, _, _ = model(pointcloud.transpose(1, 2))
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = predicted.item()

    return {
        'predicted_class': CLASSES[predicted_class],
        'processed_points': processed_pointcloud.cpu().numpy().tolist()  # 转换为Python列表
    }


if __name__ == "__main__":
    model_path = './PTH/best_pointnet10_cls_opt1.pth'
    file_path = f'{DATA_PATH}/toilet/train/toilet_0001.off'

    predicted_class = predict_single_file(file_path, model_path)
    print(f"预测类别: {predicted_class}")
