import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit, QPushButton, QMessageBox
from sympy import symbols, integrate
from collections import deque


class Calculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Calculator with Integral')

        # 设置布局为QGridLayout
        self.layout = QGridLayout()

        # 输入框，用于显示计算表达式
        self.input_line = QLineEdit()
        self.layout.addWidget(self.input_line, 0, 0, 1, 4)  # 跨4列

        # 创建按钮
        self.create_buttons()

        self.setLayout(self.layout)

    def create_buttons(self):
        # 创建九宫格按钮布局
        buttons = [
            ('1', 1, 0), ('2', 1, 1), ('3', 1, 2), ('+', 1, 3),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('-', 2, 3),
            ('7', 3, 0), ('8', 3, 1), ('9', 3, 2), ('*', 3, 3),
            ('0', 4, 0), ('C', 4, 1), ('=', 4, 2), ('/', 4, 3),
            ('∫', 5, 0, 1, 4)  # 积分按钮占据第五行的全部列
        ]

        for btn_text, row, col, rowspan, colspan in [(*btn, 1, 1) if len(btn) == 3 else btn for btn in buttons]:
            button = QPushButton(btn_text)
            button.clicked.connect(lambda _, text=btn_text: self.on_button_click(text))
            self.layout.addWidget(button, row, col, rowspan, colspan)

    def on_button_click(self, text):
        if text == "=":
            self.evaluate_expression()
        elif text == "C":
            self.input_line.clear()
        elif text == "∫":
            self.integral_operation()
        else:
            self.input_line.setText(self.input_line.text() + text)

    def evaluate_expression(self):
        try:
            expression = self.input_line.text()
            result = self.calculate_with_stack(expression)
            self.input_line.setText(str(result))
        except Exception as e:
            QMessageBox.warning(self, "Error", "Invalid expression")

    def integral_operation(self):
        try:
            expr = self.input_line.text()
            x = symbols('x')
            # 进行不定积分
            integral_result = integrate(expr, x)
            self.input_line.setText(str(integral_result))
        except Exception as e:
            QMessageBox.warning(self, "Error", "Invalid integral expression")

    def calculate_with_stack(self, expression):
        """
        使用两个栈来处理表达式，一个栈保存操作数，另一个栈保存运算符
        """
        operators = deque()
        operands = deque()
        i = 0

        while i < len(expression):
            ch = expression[i]

            # 如果当前字符是空格，跳过
            if ch == ' ':
                i += 1
                continue

            # 如果是数字，处理连续的多位数
            if ch.isdigit():
                num = 0
                while i < len(expression) and expression[i].isdigit():
                    num = num * 10 + int(expression[i])
                    i += 1
                operands.append(num)  # 将数字压入操作数栈
                continue

            # 如果是左括号，压入操作符栈
            if ch == '(':
                operators.append(ch)

            # 如果是右括号，解决括号内的表达式
            elif ch == ')':
                while operators and operators[-1] != '(':
                    self.process_operator(operators, operands)
                operators.pop()  # 弹出左括号

            # 如果是操作符
            elif ch in '+-*/':
                # 处理栈中的高优先级操作符
                while operators and self.has_precedence(ch, operators[-1]):
                    self.process_operator(operators, operands)
                operators.append(ch)  # 将当前操作符压入栈

            i += 1

        # 处理栈中剩余的操作符
        while operators:
            self.process_operator(operators, operands)

        return operands.pop()

    def process_operator(self, operators, operands):
        """
        根据栈顶的操作符和操作数进行计算，并将结果压入操作数栈
        """
        op = operators.pop()
        right = operands.pop()
        left = operands.pop()

        if op == '+':
            operands.append(left + right)
        elif op == '-':
            operands.append(left - right)
        elif op == '*':
            operands.append(left * right)
        elif op == '/':
            operands.append(left / right)

    def has_precedence(self, op1, op2):
        """
        判断操作符 op1 是否优先级比 op2 高
        """
        if op2 == '(' or op2 == ')':
            return False
        if (op1 == '*' or op1 == '/') and (op2 == '+' or op2 == '-'):
            return False
        return True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    calc = Calculator()
    calc.show()
    sys.exit(app.exec_())