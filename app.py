from flask import Flask, render_template, jsonify
import numpy as np
import os
from path import Path
from PointNet import *
from model1 import predict_single_file as predict_model1
from model2 import predict_single_file as predict_model2
from model2 import Sphere_Occlusion, RandomShift, RandomScale
from config import DATA_PATH

app = Flask(__name__)

# 添加模型路径
MODEL1_PATH = './PTH/best_pointnet10_cls.pth'
MODEL2_PATH = './PTH/best_pointnet10_cls_opt1.pth'
TRAIN_OR_TEST = 'test'


def read_off(file_path):
    with open(file_path, 'r') as file:
        if file.readline().strip() != 'OFF':
            raise ValueError('Not a valid OFF header')

        n_verts, n_faces, _ = map(int, file.readline().strip().split())

        vertices = []
        for _ in range(n_verts):
            vertex = list(map(float, file.readline().strip().split()))[:3]
            vertices.append(vertex)

        return vertices


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_point_cloud/<category>/<filename>')
def get_point_cloud(category, filename):
    try:
        file_path = Path(DATA_PATH) / category / "test" / filename

        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': f'File not found: {file_path}'
            })

        points = read_off(file_path)
        points = np.array(points)
        print(points)

        min_vals = np.min(points, axis=0).tolist()
        max_vals = np.max(points, axis=0).tolist()

        print(f"Total points: {len(points)}")
        print(f"Data range: min={min_vals}, max={max_vals}")

        return jsonify({
            'status': 'success',
            'points': points.tolist(),
            'bounds': {
                'min': min_vals,
                'max': max_vals
            }
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@app.route('/get_available_files')
def get_available_files():
    try:
        base_path = Path(DATA_PATH)
        if not os.path.exists(base_path):
            return jsonify({
                'status': 'error',
                'message': f'Base path not found: {base_path}'
            })

        categories = [d for d in os.listdir(base_path) if os.path.isdir(base_path / d)]

        files_by_category = {}
        for category in categories:
            train_path = base_path / category / TRAIN_OR_TEST
            if os.path.exists(train_path):
                files = [f for f in os.listdir(train_path) if f.endswith('.off')]
                files_by_category[category] = files

        return jsonify({
            'status': 'success',
            'categories': categories,
            'files': files_by_category
        })
    except Exception as e:
        print(f"Error in get_available_files: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@app.route('/pointnet')
def model1_page():
    return render_template('PointNet.html')


@app.route('/enhancement')
def model2_page():
    return render_template('enhancement.html')


@app.route('/predict_model/<category>/<filename>/<model>')
def predict_with_model(category, filename, model):
    try:
        file_path = Path(DATA_PATH) / category / TRAIN_OR_TEST / filename
        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': f'File not found: {file_path}'
            })

        # 获取原始点云数据
        points = read_off(file_path)
        points = np.array(points)

        # print(points)
        # print(type(points))
        # 模型预测
        if model == "model1":
            predicted_class = predict_model1(str(file_path), MODEL1_PATH)
        else:
            predicted_class = predict_model2(str(file_path), MODEL2_PATH)
        points = np.array(predicted_class['processed_points'])
        # print(points)
        # print(type(points))

        # 计算数据范围，用于前端显示
        min_vals = np.min(points, axis=0).tolist()
        max_vals = np.max(points, axis=0).tolist()

        return jsonify({
            'status': 'success',
            'points': points.tolist(),
            'classes': str(predicted_class['predicted_class']),
            'bounds': {
                'min': min_vals,
                'max': max_vals
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@app.route('/entire_process/<category>/<filename>')
def process_model2(category, filename):
    try:
        file_path = Path(DATA_PATH) / category / TRAIN_OR_TEST / filename
        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': f'File not found: {file_path}'
            })

        # 读取原始点云
        print(f"Reading file: {file_path}")  # 调试日志
        original_points = np.array(read_off(file_path))
        print(f"Original points shape: {original_points.shape}")  # 调试日志

        # 创建转换器
        transforms_sphere = Compose([
            Normalize(),
            Sphere_Occlusion()
        ])

        transforms_shift = Compose([
            Normalize(),
            Sphere_Occlusion(),
            RandomShift()
        ])

        transforms_scale = Compose([
            Normalize(),
            Sphere_Occlusion(),
            RandomShift(),
            RandomScale()
        ])

        # 应用转换
        try:
            print("Applying sphere transformation")  # 调试日志
            sphere_points = transforms_sphere(original_points)

            print("Applying shift transformation")  # 调试日志
            shift_points = transforms_shift(original_points)

            print("Applying scale transformation")  # 调试日志
            scale_points = transforms_scale(original_points)
        except Exception as e:
            print(f"Transform error: {str(e)}")  # 调试日志
            return jsonify({
                'status': 'error',
                'message': f'Transform error: {str(e)}'
            })

        # 计算每组点云的bounds
        def calculate_bounds(points):
            try:
                return {
                    'min': np.min(points, axis=0).tolist(),
                    'max': np.max(points, axis=0).tolist()
                }
            except Exception as e:
                print(f"Error calculating bounds: {str(e)}")  # 调试日志
                return None

        # 准备返回数据
        try:
            response_data = {
                'status': 'success',
                'original_points': original_points.tolist(),
                'original_bounds': calculate_bounds(original_points),
                'sphere_points': sphere_points.tolist(),
                'sphere_bounds': calculate_bounds(sphere_points),
                'shift_points': shift_points.tolist(),
                'shift_bounds': calculate_bounds(shift_points),
                'scale_points': scale_points.tolist(),
                'scale_bounds': calculate_bounds(scale_points)
            }
            return jsonify(response_data)
        except Exception as e:
            print(f"Error preparing response: {str(e)}")  # 调试日志
            return jsonify({
                'status': 'error',
                'message': f'Error preparing response: {str(e)}'
            })

    except Exception as e:
        print(f"General error: {str(e)}")  # 调试日志
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True)
