import tkinter as tk

class MyGUI():
    window = tk.Tk()  # 生成一个窗口对象
    window.title('我的GUI')  # 设置窗口标题

    # 第一组按钮
    frame1 = tk.Frame(window)  # 生成第一组按钮的容器
    frame1.grid(row=0, column=0, sticky='w')  # sticky='w'指定了组件在单元格中靠左对齐
    tk.Label(frame1, text='第一组').pack(side='left')  # 添加本组标题
    tk.Button(frame1, text="按钮1").pack(side='left')  # 添加按钮
    tk.Button(frame1, text="按钮2").pack(side='left')  # 添加按钮
    tk.Button(frame1, text="按钮3").pack(side='left')  # 添加按钮

    # 第二组按钮
    frame2 = tk.Frame(window)  # 生成第一组按钮的容器
    frame2.grid(row=1, column=0, sticky='w')  # sticky='w'指定了组件在单元格中靠左对齐
    tk.Label(frame2, text='第二组').pack(side='left')  # 添加本组标题
    tk.Button(frame2, text="按钮4").pack(side='left')  # 添加按钮
    tk.Button(frame2, text="按钮5").pack(side='left')  # 添加按钮

    # 第三组按钮
    frame3 = tk.Frame(window)  # 生成第一组按钮的容器
    frame3.grid(row=2, column=0, sticky='nw')  # sticky='w'指定了组件在单元格中靠左对齐
    tk.Label(frame3, text='第三组').pack(side='left')  # 添加本组标题
    tk.Button(frame3, text="按钮6").pack(side='left')  # 添加按钮
    tk.Button(frame3, text="按钮7").pack(side='left')  # 添加按钮

    window.mainloop()  # 创建事件循环（不必理解，照抄即可）

if __name__ == '__main__':
    MyGUI() # 启动GUI