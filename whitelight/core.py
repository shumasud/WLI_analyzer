import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal
import math


class EnvelopePeak(object):
    def __init__(self, x, y, ep0):
        self.offset = 0
        self.peak = [x[ep0], y[ep0]]
        self._x = x
        self._y = y
        self._find_peak(ep0)

    def _find_peak(self, ep0, f_rate=0.5):
        """
        包絡線ピークのインデックスを求める(二乗＋ローパスにより包絡線を求める）
        緊急作業につき，今後’絶対’修正する
        """

        # フィッティングする関数
        def gaussian(xx, a, b, c):
            yy = a * np.exp(-((xx - b) ** 2) / (2 * c * c))
            return yy

        #   フィッティング範囲を決定
        threshold = f_rate * self.peak[1]
        fit_range = 0
        for i in range(ep0, len(self._y)):
            if self._y[i] < threshold:
                fit_range = i - ep0
                break
        xx = self._x[(ep0 - fit_range): (ep0 + fit_range)]
        yy = self._y[(ep0 - fit_range): (ep0 + fit_range)]
        self.offset = ep0 - fit_range
        sigma = np.max(xx) - self.peak[0]

        #   フィッティング
        try:
            coef, pconv = curve_fit(gaussian, xx, yy, p0=[self.peak[1], self.peak[0], sigma])
        except RuntimeError:
            print("fail to fit on ", *self.peak)
        else:
            #   フィッティング結果から頂点のx座標と包絡線を保存
            self.peak = (coef[1], gaussian(coef[1], *coef))
            self._y = list(map(lambda x: gaussian(x, *coef), xx))
            self._x = xx
        print(self)

    def __str__(self):
        return 'peak[um] ({:.3f}, {:.3f})'.format(self.peak[0], self.peak[1])

    def show(self, ax=None):
        if ax:
            ax.plot(self._x, self._y)
            ax.plot(self.peak[0], self.peak[1], 'o')
        # print(self._x, self._y)
        return


class Fringes(object):
    def __init__(self, x=[0], y=[0], fs=1, **detail):
        """

        Parameters
        ----------
        x
        y
        fs
        detail
        """
        self.x = x
        self.y = y
        self.fs = fs
        self.env = []
        self.peaks = []
        self.detail = detail

    def find_peaks(self, threshold=0.5, cutoffrate=50):
        #   包絡線極大値のインデックスのリストを求める
        self.env = lpf( self.y**2, self.fs, self.fs/cutoffrate )
        relmaxs = signal.argrelmax(self.env, order=1)[0]
        #   閾値を越えた極大値のみ処理
        threshold = self.env.max() * threshold
        self.peaks = []
        self.env = np.abs(signal.hilbert(self.y))
        for relmax in relmaxs:
            if threshold < self.env[relmax]:
                ep = EnvelopePeak(self.x, self.env, relmax)
                self.peaks.append(ep)

    def down_sample(self, step):
        self.x = self.x[::step]
        self.y = self.y[::step]

    def show(self, ax=None):
        if ax:
            ax.plot(self.x, self.y)
            ax.plot(self.x, self.env)
            ax.grid(which='major', color='black', linestyle='-')

            for ep in self.peaks:
                ep.show(ax)
            ax.legend(["fringe", "envelope", "fitting"])


class WhiteLight(object):
    """
    白色干渉波形の解析（単純なマイケルソン干渉計によるスキャニングを想定）

    Parameters
        fringe : array
            干渉縞データ
        fs : float
            データのサンプリング周波数

    Attributes
        fringe_sq_ : array
            白色干渉縞の二乗信号
        envelope_ : array
            包絡線

    """

    def spe_ana(self, wave, axis):
        """
        スペクトルアナライザー（スペクトルをプロット）

        Parameters
            wave : array
                信号

        """
        number = len(wave)
        spectrum = fftpack.fft(wave)
        frecency = [abs(k * self.fs / number) for k in range(number)]
        axis.plot(frecency, abs(spectrum))
        axis.set_xlim([0, np.max(frecency) / 2])
    def __init__(self, fringe, fs):
        self.fringe = fringe
        self.fs = fs
        #   干渉縞を二乗
        self.fringe_sq_ = self.fringe * self.fringe


    def calc_EPs(self, cof_env, ep_sens, method='SL', f_rate=0.5):
        """
        包絡線ピークのインデックスを求める(二乗＋ローパスにより包絡線を求める）
        
        Parameters
            cof_env : float
                包絡線を求める際のローパスフィルタのカットオフ周波数
            ep_sens : float
                包絡線のピーク検知の感度（これ以上の極大値をピークとして認識）
        
        Attributes
            fringe_sq_ : array
                白色干渉縞の二乗
            envelope_ : 
        
        Returns
            eps : list
                包絡線ピークのインデックスのリスト
        
        """
        """SL法での包絡線を求め、その極大値のインデックスを記録"""
        #   ローパスフィルタをかけ、包絡線を求める
        self.envelope_ = lpf(self.fringe_sq_, self.fs, cof_env)
        #   包絡線極大値のインデックスのリストを求める
        env_relmax = signal.argrelmax(self.envelope_)[0]

        """SL法のときはそのまま包絡線ピークとして採用、HG法ではさらに計算"""
        if method == 'SL':
            """真の包絡線ピークを決定"""
            #   真の包絡線ピーク(SL法)のリスト
            env_relmax_true = []
            #   値が感度より高いものを選定
            for i in range(len(env_relmax)):
                if self.envelope_[env_relmax[i]] >= ep_sens:
                    env_relmax_true.append(env_relmax[i])
            eps = env_relmax_true

        elif method == 'HG':

            def gaussian(xx, a, b, c):
                """理論的な包絡線（ガウシアン）"""
                yy = a * np.exp(-((xx - b) ** 2) / (2 * c * c))
                return yy

            #   ヒルベルト変換により包絡線を求める
            self.envelope_ = np.abs(signal.hilbert(self.fringe))

            """真の包絡線ピークを決定"""
            #   真の包絡線ピーク(SL法)のリスト
            env_relmax_true = []
            #   値が感度より高いものを選定
            for i in range(len(env_relmax)):
                if self.envelope_[env_relmax[i]] >= ep_sens:
                    env_relmax_true.append(env_relmax[i])

            eps = []
            for ep in env_relmax_true:
                #   フィッティング範囲を決定
                for i in range(ep, len(self.envelope_)):
                    if self.envelope_[i] < f_rate * self.envelope_[ep]:
                        fit_range = i - ep
                        break
                #   フィッティング範囲のx座標
                x = np.arange(ep - fit_range, ep + fit_range)
                #   フィッティング範囲のy座標
                y = self.envelope_[ep - fit_range: ep + fit_range]
                #   フィッティングの初期値
                initial = [self.envelope_[ep], ep, fit_range]
                #   フィッティング
                coef, pconv = curve_fit(gaussian, x, y, p0=initial)
                #   フィッティングでの頂点のx座標を追加
                eps.append((coef[1]))
                #   フィッティング後の包絡線をプロット
                envelope = [gaussian(i, coef[0], coef[1], coef[2]) for i in x]
                fig_gauss = plt.figure(1)
                ax = fig_gauss.add_subplot(111)
                ax.plot(x, y, "ro", markersize=2)
                ax.plot(x, envelope, linewidth=3)
        else:
            print('定義されてない手法です')
            # sys.exit()
        return eps


class Sest(object):
    """
    SESTアルゴリズムの実装

    Parameters
        X ; array
            干渉縞の標本データ(y座標)
        Y : array
            干渉縞の標本データ(x座標)
        lambda_c : float
            光源の中心波長
        wl_hw : float
            光源のHWHM
        delta : float
            サンプリング間隔

    """

    def __init__(self, X, Y, wl_c, wl_hw, delta):
        self.X = X
        self.Y = Y
        self.wl_c = wl_c
        self.wl_hw = wl_hw
        self.delta = delta

    def reconstruction(self, x):
        """
        干渉縞の復元処理

        Parameters
            x : float
                復元点のx座標

        Returns
            y : float
                xでの復元後の干渉縞のy座標
        """
        d = np.array([1 / 2 / self.delta])
        k = 4 * np.pi / self.wl_c
        lst = []
        for i, (x_n, y_n) in enumerate(zip(self.X, self.Y)):
            lst.append(y_n * np.sinc((x - x_n) * d) * np.cos((x - x_n) * k))
        print(x)
        y = np.sum(lst)
        return y

    def decide_srate(self):
        """
        許されるサンプリング間隔の可視化

        """
        max_I = (self.wl_c + self.wl_hw) / (2 * self.wl_hw)
        I = []
        for i in range(int(max_I + 1)):
            I.append(i)
        I.pop(0)
        min_delta = []
        max_delta = []
        for i in I:
            min_delta.append((i - 1) * (self.wl_c + self.wl_hw) / 4)
            max_delta.append(i * (self.wl_c - self.wl_hw) / 4)
        for i in range(len(min_delta)):
            plt.plot([min_delta[i], max_delta[i]], [1, 1], color='b')
            plt.plot((min_delta[i] + max_delta[i]) / 2, 1, marker='o', color='r')


def lpf(f, fs, fc):
    """
    ローパスフィルター

    Parameters
        f : array
            元の信号
        fs : float
            サンプリング周波数
        fc : float
            カットオフ周波数

    Returns
        y : array
            カットオフ後の信号
    """
    freq = np.linspace(0, fs, len(f))
    y = np.fft.fft(f)
    y[ fc<freq ] = 0
    y[0] = y[0] / 2
    y = np.real( np.fft.ifft(y*2) )
    return y

def bpf(x, fs, fe1, fe2):
    """
    バンドパスフィルター

    Parameters
        x : array
            元の信号
        fs : float
            サンプリング周波数
        fe0 : float
            下側カットオフ周波数
        fe1 : float
            上側カットオフ周波数

    Returns
        y : array
            カットオフ後の信号
    """
    X = fftpack.fft(x)
    frecency = [abs(k * fs / len(x)) for k in range(len(x))]
    for k, f in enumerate(frecency):
        if (fe2 < f < frecency[-1] - fe2) == 1:
            X[k] = complex(0, 0)
        elif (fe1 < f < frecency[-1] - fe1) == 0:
            X[k] = complex(0, 0)
    y = np.real(fftpack.ifft(X))
    return y

