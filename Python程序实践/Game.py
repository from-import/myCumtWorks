import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QMessageBox, QPushButton, QFileDialog, \
    QVBoxLayout, QHBoxLayout, QDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
import random


class PuzzlePiece(QLabel):
    def __init__(self, pixmap, position):
        super().__init__()
        self.setPixmap(pixmap)
        self.position = position  # 记录拼图块的初始位置
        self.setFixedSize(pixmap.size())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parentWidget().move_piece(self.position)


class PuzzleGame(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_size = 3  # 默认 3x3 的拼图
        self.puzzle_size = 600  # 拼图区域的大小
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Puzzle Game')
        self.setGeometry(100, 100, 800, 800)  # 扩大窗口以确保按钮显示

        # 主布局
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # 创建网格布局
        self.grid_layout = QGridLayout()
        self.main_layout.addLayout(self.grid_layout)

        # 按钮布局
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)

        # 按钮
        self.original_button = QPushButton('查看原图')
        self.shuffle_button = QPushButton('随机切换图片')
        self.start_button = QPushButton('开始挑战')
        self.difficulty_button = QPushButton('切换难度')

        self.button_layout.addWidget(self.original_button)
        self.button_layout.addWidget(self.shuffle_button)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.difficulty_button)

        self.original_button.clicked.connect(self.view_original)
        self.shuffle_button.clicked.connect(self.shuffle_image)
        self.start_button.clicked.connect(self.start_challenge)
        self.difficulty_button.clicked.connect(self.change_difficulty)

        # 加载图片并进行分割
        self.load_and_scale_image('test.png')
        self.pieces = []
        self.empty_position = (self.grid_size - 1, self.grid_size - 1)  # 最后一个格子为空

        self.challenge_timer = QTimer(self)
        self.challenge_timer.timeout.connect(self.time_out)
        self.challenge_time = 60  # 挑战时间，默认60秒
        self.time_remaining = self.challenge_time

        self.create_pieces()
        self.shuffle_pieces()

        self.show()

    def load_and_scale_image(self, image_path):
        """加载并统一缩放图片到拼图区域大小"""
        self.original_pixmap = QPixmap(image_path)
        self.original_pixmap = self.original_pixmap.scaled(self.puzzle_size, self.puzzle_size, Qt.KeepAspectRatio)
        self.piece_size = self.original_pixmap.width() // self.grid_size

    def create_pieces(self):
        """根据当前难度创建拼图块"""
        self.clear_pieces()  # 清空拼图块
        self.pieces.clear()
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) == self.empty_position:
                    continue  # 空白区域
                piece_pixmap = self.original_pixmap.copy(col * self.piece_size, row * self.piece_size, self.piece_size,
                                                         self.piece_size)
                piece = PuzzlePiece(piece_pixmap, (row, col))
                self.pieces.append((piece, row, col))
                self.grid_layout.addWidget(piece, row, col)

    def clear_pieces(self):
        """清空当前拼图块"""
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                self.grid_layout.removeWidget(widget)
                widget.deleteLater()

    def shuffle_pieces(self):
        """随机打乱拼图块"""
        random.shuffle(self.pieces)
        for idx, (piece, row, col) in enumerate(self.pieces):
            r, c = divmod(idx, self.grid_size)
            self.grid_layout.addWidget(piece, r, c)
            self.pieces[idx] = (piece, r, c)
            piece.position = (r, c)

    def move_piece(self, position):
        """点击移动拼图块"""
        row, col = position
        if self.is_adjacent_to_empty(row, col):
            # 找到点击的拼图块
            for idx, (piece, r, c) in enumerate(self.pieces):
                if (r, c) == (row, col):
                    # 将拼图块移动到空白位置
                    self.grid_layout.addWidget(piece, *self.empty_position)
                    piece.position = self.empty_position
                    self.pieces[idx] = (piece, *self.empty_position)
                    self.empty_position = (row, col)
                    break

            if self.check_win():
                QMessageBox.information(self, "Congratulations!", "You completed the puzzle!")
                self.challenge_timer.stop()

    def is_adjacent_to_empty(self, row, col):
        """检查是否与空白块相邻"""
        empty_row, empty_col = self.empty_position
        return abs(empty_row - row) + abs(empty_col - col) == 1

    def check_win(self):
        """检查拼图是否完成"""
        for idx, (piece, row, col) in enumerate(self.pieces):
            original_row, original_col = divmod(idx, self.grid_size)
            if row != original_row or col != original_col:
                return False
        return True

    def view_original(self):
        """查看原图 - 弹出图片框显示原图"""
        original_dialog = QDialog(self)
        original_dialog.setWindowTitle("原始图片")
        original_dialog.setGeometry(100, 100, self.puzzle_size, self.puzzle_size)

        original_label = QLabel(original_dialog)
        original_label.setPixmap(self.original_pixmap)
        original_label.setGeometry(0, 0, self.puzzle_size, self.puzzle_size)

        original_dialog.exec_()

    def shuffle_image(self):
        """随机切换图片"""
        file_dialog = QFileDialog()
        file_dialog.setNameFilters(["Images (*.png *.xpm *.jpg)"])
        if file_dialog.exec_():
            image_path = file_dialog.selectedFiles()[0]
            self.load_and_scale_image(image_path)
            self.create_pieces()
            self.shuffle_pieces()

    def change_difficulty(self):
        """切换拼图难度"""
        difficulties = [(3, '简单'), (4, '中等'), (5, '困难')]
        current_idx = next(i for i, (n, _) in enumerate(difficulties) if n == self.grid_size)
        next_idx = (current_idx + 1) % len(difficulties)
        self.grid_size, difficulty_name = difficulties[next_idx]

        self.piece_size = self.original_pixmap.width() // self.grid_size
        self.empty_position = (self.grid_size - 1, self.grid_size - 1)
        self.create_pieces()
        self.shuffle_pieces()
        QMessageBox.information(self, "Difficulty Changed", f"难度已设置为 {difficulty_name}.")

    def start_challenge(self):
        """开始挑战模式"""
        self.time_remaining = self.challenge_time
        self.challenge_timer.start(1000)  # 每秒倒计时
        QMessageBox.information(self, "Challenge", "挑战开始！你有60秒的时间完成拼图。")

    def time_out(self):
        """挑战时间结束"""
        self.time_remaining -= 1
        if self.time_remaining <= 0:
            self.challenge_timer.stop()
            QMessageBox.information(self, "Time's Up", "挑战失败！时间已到。")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = PuzzleGame()
    sys.exit(app.exec_())