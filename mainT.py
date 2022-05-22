from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from optimization_func import *
import numpy as np
xx=None
error=None
def generate_scatter_chart():
    global canvas1
    global canvas2
    global error
    if canvas1:
        canvas1.get_tk_widget().destroy()
    if canvas2:
        canvas2.get_tk_widget().destroy()
    fig = Figure(figsize=(20, 20), dpi=200)
    ax = fig.add_subplot(111)
    filename = '5.sad'
    h_t = np.loadtxt(filename)
    print("attempt")
    global xx
    print(xx)
    xx, X, y,error = optimization_circle(h_t,int(grade[power.get()]), xx=xx)
    print("attempt2")
    ax.plot(X, y,label="Полученная характеристика")
    canvas1 = FigureCanvasTkAgg(fig, master=window)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=TOP, fill=NONE, expand=0)
    window.after(200, None)

    ax.plot(h_t[:, 0], h_t[:, 1],label="Исходный график")
    canvas2 = FigureCanvasTkAgg(fig, master=window)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=TOP, fill=NONE, expand=0)
    ax.grid()
    ax.legend()
    window.after(200, None)



def show_params():
    global xx
    xx /= max(xx)
    new_window = Toplevel()
    new_window.title('Params')
    new_window.geometry("700x700")
    for index in range(xx.shape[0]):
        print(xx[index])
        if(index<xx.shape[0]/2):
            labelN = Label(new_window,
            text="a"+str(index+1)+"  -  "+str(round(xx[index],4)))
        else:
            labelN = Label(new_window,
            text="b" + str(int(abs(xx.shape[0]/2-index)+1)) + "  -  " + str(round(xx[index], 4)))
        labelN.pack()
    label5 = Label(new_window,
                   text="Результурующая ошибка = "+ str(round(error,4)))
    label5.pack()


def close_scatter_chart():
    global canvas1
    global canvas2
    if canvas1:
        canvas1.get_tk_widget().destroy()
    if canvas2:
        canvas1.get_tk_widget().destroy()


window = Tk()
canvas1 = None
canvas2 = None
window.title('Plotting in Tkinter')
window.geometry("1000x1000")
label1 = Label(window, text="Идентификация численными методами оптимизации")
label1.pack()
btn = Button(
    master=window,
    text='Начать оптимизацию и постройку графика',
    command=generate_scatter_chart,
    padx=5, pady=5
)
btn.pack()
btn3 = Button(
    master=window,
    text='Посмотреть параметры',
    command=show_params,
    padx=5, pady=5
)
btn3.pack()
btn2 = Button(
    master=window,
    text='Закрыть график',
    command=close_scatter_chart,
    padx=5, pady=5
)
btn2.pack()
grade = ["2", "3"]
power = IntVar()
label2 = Label(window, text="Выберите степень полинома")
label2.pack()
for index in range(len(grade)):
    radiobutton = Radiobutton(window,
                              text=grade[index],
                              variable=power,
                              value=index)
    radiobutton.pack()



window.mainloop()
