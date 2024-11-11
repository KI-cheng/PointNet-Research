import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,QHBoxLayout, QComboBox, QPushButton, QLabel, QFileDialog)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path

"""
该代码已废弃，open3d运行问题很大（总有神奇bug），改用matplotlib和qt5卡得要死，还是flask吧
"""
class PointCloudViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Point Cloud Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 创建控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)

        # 添加类别选择控件
        self.category_combo = QComboBox()
        self.category_combo.currentIndexChanged.connect(self.on_category_changed)
        control_layout.addWidget(QLabel("Category:"))
        control_layout.addWidget(self.category_combo)

        # 添加文件选择控件
        self.file_combo = QComboBox()
        control_layout.addWidget(QLabel("File:"))
        control_layout.addWidget(self.file_combo)

        # 添加加载按钮
        load_button = QPushButton("Load Point Cloud")
        load_button.clicked.connect(self.load_point_cloud)
        control_layout.addWidget(load_button)

        # 添加保存按钮
        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_figure)
        control_layout.addWidget(save_button)

        # 添加状态标签
        self.status_label = QLabel()
        control_layout.addWidget(self.status_label)

        # 添加视角按钮
        view_layout = QVBoxLayout()
        views = [
            ("Top View", (90, 0)),
            ("Front View", (0, 0)),
            ("Side View", (0, 90)),
            ("Isometric View", (45, 45))
        ]
        for view_name, angles in views:
            btn = QPushButton(view_name)
            btn.clicked.connect(lambda checked, a=angles: self.set_view_angle(*a))
            view_layout.addWidget(btn)
        control_layout.addLayout(view_layout)

        # 间隔
        control_layout.addStretch()

        # matplotlib画布
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 图形布局
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)

        # 面板和图形添加到主布局
        layout.addWidget(control_panel)
        layout.addLayout(plot_layout)

        # 初始化
        self.points = None
        self.setup_initial_data()

    def save_figure(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Figure",
                "",
                "PNG Files (*.png);;All Files (*)"
            )
            if file_path:
                self.fig.savefig(file_path, bbox_inches='tight', dpi=300)
                self.status_label.setText(f"Saved to {file_path}")
        except Exception as e:
            self.status_label.setText(f"Error saving figure: {str(e)}")

    def setup_initial_data(self):
        try:
            base_path = Path("./ModelNet10")
            if not base_path.exists():
                self.status_label.setText("Error: Dataset not found")
                return

            categories = [d for d in os.listdir(base_path) if os.path.isdir(base_path / d)]
            self.category_combo.addItems(categories)

            self.base_path = base_path
            self.update_file_list()

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def update_file_list(self):
        try:
            category = self.category_combo.currentText()
            if not category:
                return

            train_path = self.base_path / category / "train"
            files = [f for f in os.listdir(train_path) if f.endswith('.off')]

            self.file_combo.clear()
            self.file_combo.addItems(files)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def on_category_changed(self): # 类别改变时更新文件列表
        self.update_file_list()

    def read_off(self, file_path):
        try:
            with open(file_path, 'r') as file:
                if file.readline().strip() != 'OFF':
                    raise ValueError('Not a valid OFF header')

                n_verts, n_faces, _ = map(int, file.readline().strip().split())

                vertices = []
                for _ in range(n_verts):
                    vertex = list(map(float, file.readline().strip().split()))[:3]
                    vertices.append(vertex)

                return np.array(vertices)
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    def load_point_cloud(self): # 加载点云，直接显示点云
        try:
            category = self.category_combo.currentText()
            filename = self.file_combo.currentText()

            if not category or not filename:
                self.status_label.setText("Please select category and file")
                return

            file_path = self.base_path / category / "train" / filename
            self.points = self.read_off(file_path)

            self.plot_point_cloud()
            self.status_label.setText(f"Loaded {len(self.points)} points")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def plot_point_cloud(self): # 对点云进行绘制
        if self.points is None:
            return
        self.ax.clear()

        self.ax.scatter(self.points[:, 0],
                        self.points[:, 1],
                        self.points[:, 2],
                        c='g',  # 使用绿色
                        marker='.',
                        s=1)

        # 标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # 设置坐标轴比例相等
        max_range = np.array([
            self.points[:, 0].max() - self.points[:, 0].min(),
            self.points[:, 1].max() - self.points[:, 1].min(),
            self.points[:, 2].max() - self.points[:, 2].min()
        ]).max() / 2.0

        mid_x = (self.points[:, 0].max() + self.points[:, 0].min()) * 0.5
        mid_y = (self.points[:, 1].max() + self.points[:, 1].min()) * 0.5
        mid_z = (self.points[:, 2].max() + self.points[:, 2].min()) * 0.5

        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 设置网格
        self.ax.grid(True)

        # 刷新画布
        self.canvas.draw()

    def set_view_angle(self, elev, azim):
        if self.points is not None:
            self.ax.view_init(elev=elev, azim=azim)
            self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    viewer = PointCloudViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()