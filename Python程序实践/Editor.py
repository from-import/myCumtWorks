import sys
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QMdiArea, QMdiSubWindow, QApplication, QAction, QFileDialog,
                             QFontDialog, QVBoxLayout, QWidget, QToolBar, QComboBox)
from PyQt5.QtGui import QFont

class SimpleMDIExample(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        self.setWindowTitle('SimpleMDIExample')

        self.create_actions()
        self.create_toolbar()
        self.show()

    def create_actions(self):
        # 新建文件
        new_action = QAction("New", self)
        new_action.triggered.connect(self.new_file)
        # 打开文件
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        # 保存文件
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_file)
        # 设置字体
        font_action = QAction("Set Font", self)
        font_action.triggered.connect(self.set_font)

        # 添加到菜单栏
        self.menuBar().addAction(new_action)
        self.menuBar().addAction(open_action)
        self.menuBar().addAction(save_action)
        self.menuBar().addAction(font_action)

    def create_toolbar(self):
        # 工具栏：添加字体、粗体、斜体、下划线
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        bold_action = QAction('Bold', self)
        bold_action.triggered.connect(self.set_bold)
        toolbar.addAction(bold_action)

        italic_action = QAction('Italic', self)
        italic_action.triggered.connect(self.set_italic)
        toolbar.addAction(italic_action)

        underline_action = QAction('Underline', self)
        underline_action.triggered.connect(self.set_underline)
        toolbar.addAction(underline_action)

    def new_file(self):
        sub_window = QMdiSubWindow()
        text_edit = QTextEdit()
        sub_window.setWidget(text_edit)
        sub_window.setWindowTitle('New Document')
        self.mdi.addSubWindow(sub_window)
        sub_window.show()

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt)")
        if file_name:
            with open(file_name, 'r') as f:
                file_content = f.read()
            sub_window = QMdiSubWindow()
            text_edit = QTextEdit()
            text_edit.setText(file_content)
            sub_window.setWidget(text_edit)
            sub_window.setWindowTitle(file_name)
            self.mdi.addSubWindow(sub_window)
            sub_window.show()

    def save_file(self):
        active_subwindow = self.mdi.activeSubWindow()
        if active_subwindow:
            text_edit = active_subwindow.widget()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt)")
            if file_name:
                with open(file_name, 'w') as f:
                    f.write(text_edit.toPlainText())

    def set_font(self):
        active_subwindow = self.mdi.activeSubWindow()
        if active_subwindow:
            font, ok = QFontDialog.getFont()
            if ok:
                text_edit = active_subwindow.widget()
                text_edit.setFont(font)

    def set_bold(self):
        active_subwindow = self.mdi.activeSubWindow()
        if active_subwindow:
            text_edit = active_subwindow.widget()
            current_font = text_edit.currentFont()
            # 切换粗体：如果当前是粗体，则取消；否则启用
            current_font.setBold(not current_font.bold())
            text_edit.setCurrentFont(current_font)

    def set_italic(self):
        active_subwindow = self.mdi.activeSubWindow()
        if active_subwindow:
            text_edit = active_subwindow.widget()
            current_font = text_edit.currentFont()
            # 切换斜体：如果当前是斜体，则取消；否则启用
            current_font.setItalic(not current_font.italic())
            text_edit.setCurrentFont(current_font)

    def set_underline(self):
        active_subwindow = self.mdi.activeSubWindow()
        if active_subwindow:
            text_edit = active_subwindow.widget()
            current_font = text_edit.currentFont()
            # 切换下划线：如果当前有下划线，则取消；否则启用
            current_font.setUnderline(not current_font.underline())
            text_edit.setCurrentFont(current_font)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = SimpleMDIExample()
    sys.exit(app.exec_())