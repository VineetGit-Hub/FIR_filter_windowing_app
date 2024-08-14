import tkinter as tk
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal.windows import kaiser

class AnimatedPlotApp:
    def __init__(self, master):
        self.master = master
        master.title("Animated Plot App")

        # GUI elements for input parameters
        self.param_label = tk.Label(master, text="Enter Parameters:")
        self.param_label.pack()

        label_widget = tk.Label(master, text='N (window length)')
        label_widget.pack()
        self.N = tk.Entry(master)
        self.N.insert(0, 10)
        self.N.pack()

        label_widget = tk.Label(master, text='w_c (ILPF cut-off frequency)')
        label_widget.pack()
        self.w_c = tk.Entry(master)
        self.w_c.insert(0, 1)
        self.w_c.pack()

        label_widget = tk.Label(master, text='Frame interval')
        label_widget.pack()
        self.interval = tk.Entry(master)
        self.interval.insert(0, 1)
        self.interval.pack()

        label_widget = tk.Label(master, text='Window type')
        label_widget.pack()
        self.window_type = tk.Entry(master)
        self.window_type.insert(0, 'rect bspline2 hamm0.54 kaiser14')
        self.window_type.pack()

        self.start_button = tk.Button(master, text="Start Animation", command=self.start_animation)
        self.start_button.pack()


    def start_animation(self):
        # Call your animation function with the input parameters
        N = self.N.get().split()
        w_c = float(self.w_c.get())
        interval = float(self.interval.get())
        window_type = self.window_type.get().split()
        if len(N)==len(window_type)==1:
            self.cases = 1
            self.animate([int(N[0])], w_c, interval, window_type)
        else:
            self.cases = max(len(N), len(window_type))

            N = [int(i) for i in N]

            N += N[-1:]*(self.cases-len(N))
            window_type += window_type[-1:]*(self.cases-len(window_type))

            self.animate(N, w_c, interval, window_type)


    def get_window(self, N_, window_type_, w):
        windows = []
        for N, window_type in zip(N_, window_type_):
            window_type = window_type.strip().lower()
            if window_type.startswith('rect'):
                window = np.sin((N+0.5)*w)/np.sin(0.5*w)
            elif window_type.startswith('bspline'):
                alpha = float(window_type[7:])
                window = (np.sin((N/alpha+0.5)*w)/np.sin(0.5*w))**alpha*(alpha/(2*N))**(alpha-1)
            elif window_type.startswith('hamm'):
                alpha = float(window_type[4:])
                window = (freqz(alpha+(1-alpha)*np.cos(np.pi/N*np.arange(-N, N+1)), worN=w)[1]*np.exp(1j*N*w)).real
            elif window_type.startswith('kaiser'):
                window = (freqz(kaiser(2*N+1, float(window_type[6:])), worN=w)[1]*np.exp(1j*N*w)).real
            else:
                return
            windows.append(window)
        return np.vstack(windows)


    def animate(self, N, w_c, interval, window_type):
        dw = 0.01
        w = np.arange(-np.pi, np.pi, dw)
        ideal_lpf = np.abs(w)<=w_c
        w_c_region = w[ideal_lpf]

        fig, ax = plt.subplots(2, figsize=(13, 7))
        ax[0].set_xlim(-np.pi, np.pi)
        ax[1].set_xlim(-np.pi, np.pi)
        ax[0].set_xlabel(r'$\omega$')
        ax[1].set_xlabel(r'$\omega$')
        ax[0].set_ylabel(r'$V\left(e^{j\omega}\right)$')
        ax[1].set_ylabel(r'$H_{FIR}\left(e^{j\omega}\right)$')
        ax[0].grid()
        ax[1].grid()
        ax[0].axvspan(-w_c, w_c, color='y', alpha=0.1)
        if self.cases==1:
            fb00 = ax[0].fill_between([], [], color='b')
            fb01 = ax[0].fill_between([], [], color='r')

        window = self.get_window(N, window_type, w)

        y_min, y_max = np.min(window), np.max(window)
        y_range = (y_max-y_min)*0.05
        ax[0].set_ylim(y_min-y_range, y_max+y_range)

        conv = np.vstack([dw/(2*np.pi)*np.convolve(ideal_lpf, wind, mode='same') for wind in window])
        window = np.roll(window, window.shape[-1]//2, axis=-1)
        label_legend = (window_type[0]+', N = '+str(N[0])) if self.cases==1 else [j+', N = '+str(i) for i, j in zip(N, window_type)]
        line0 = ax[0].plot(w, window.T, label=label_legend)
        line1 = ax[1].plot(w[:0], conv[:, :0].T, label=label_legend)

        y_min, y_max = np.min(conv), np.max(conv)
        y_range = (y_max-y_min)*0.05
        ax[1].set_ylim(y_min-y_range, y_max+y_range)

        def update(t):
            nonlocal window
            window = np.roll(window, 1, axis=-1)

            for i in range(len(window)):
                line0[i].set_ydata(window[i])
                line1[i].set_data(w[:t], conv[i, :t])

            if self.cases==1:
                temp = np.vstack([w_c_region, (np.clip(window[0][ideal_lpf], a_min=0, a_max=None))]).T
                temp2 = temp.copy()
                temp2[:, 1] = 0
                temp = np.vstack([temp, temp2[::-1]])
                fb00.set_paths([temp])

                temp = np.vstack([w_c_region, (np.clip(window[0][ideal_lpf], a_min=None, a_max=0))]).T
                temp2 = temp.copy()
                temp2[:, 1] = 0
                temp = np.vstack([temp, temp2[::-1]])
                fb01.set_paths([temp])

                return line0+line1+[fb00, fb01]
            
            return line0+line1

        ani = FuncAnimation(fig, update, frames=window.shape[-1], interval=interval, blit=True, repeat=True)

        plt.legend()
        plt.show()


# Create the GUI application
root = tk.Tk()
app = AnimatedPlotApp(root)
root.mainloop()
