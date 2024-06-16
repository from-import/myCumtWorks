class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __str__(self):
        return f"{self.real} + {self.imag}i"

    def __add__(self, other):
        real_part = self.real + other.real
        imag_part = self.imag + other.imag
        return ComplexNumber(real_part, imag_part)

    def __sub__(self, other):
        real_part = self.real - other.real
        imag_part = self.imag - other.imag
        return ComplexNumber(real_part, imag_part)

    def __mul__(self, other):
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return ComplexNumber(real_part, imag_part)

    def __truediv__(self, other):
        denominator = other.real ** 2 + other.imag ** 2
        real_part = (self.real * other.real + self.imag * other.imag) / denominator
        imag_part = (self.imag * other.real - self.real * other.imag) / denominator
        return ComplexNumber(real_part, imag_part)



if __name__ == "__main__":
    c1 = ComplexNumber(3, 4)  # 3 + 4i
    c2 = ComplexNumber(1, 2)  # 1 + 2i

    print(f"+: {c1} + {c2} = {c1 + c2}")
    print(f"-: {c1} - {c2} = {c1 - c2}")
    print(f"*: {c1} * {c2} = {c1 * c2}")
    print(f"/: {c1} / {c2} = {c1 / c2}")