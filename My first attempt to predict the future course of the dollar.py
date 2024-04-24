import numpy
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

## Создание шрифтов для обозначений
font1 = {'family':'serif','color':'green','size':15}
font2 = {'family':'serif','color':'darkred','size':12}

## Создание графиков
## Даты
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
## Курс
y = [92.3660, 92.2914, 92.5254, 92.3892, 92.3058, 92.4155, 92.4155, 92.4155, 92.73, 92.7463, 93.2198, 93.7196, 93.4419, 93.4419, 93.4419]

## Создание обозначений по x и по y
plt.title("Курс доллара(данные на апрель)", fontdict = font1, loc = "left")
plt.xlabel("Дата", fontdict = font2)
plt.ylabel("Цена за 1 доллар", fontdict = font2)

## Создание полиномиальной модели
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

## Отображение строки (начальная дата, конечная дата и количество значений в premove-линии)
myline = numpy.linspace(1, 20, 2)

## Вероятность выгодного исхода
print("Вероятность того, что данные верны:", round(r2_score(y, mymodel(x)), 4))

## Исходная диаграмма
plt.plot(x, y)
## Отображение линии полиномиальной регрессии
plt.plot(myline, mymodel(myline))
## Отображение диаграммы
plt.show()