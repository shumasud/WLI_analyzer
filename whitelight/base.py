# -*- coding: utf-8 -*-
"""
Name:
    base.py
Purpose:
    干渉データを解析してpeaks distanceを計算
Specification:
    モジュール
Environment:
    Python 3.5.1
    
"""
import numpy as np
from scipy import signal
from scipy import fftpack
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
import math

class Envelope(object):
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
        fit_range = 0
        for i in range(ep0, len(self._y)):
            if self._y[i] < f_rate * self._y[ep0]:
                fit_range = i - ep0
                break
        self.offset = ep0-fit_range
        xx = self._x[ (ep0-fit_range) : (ep0+fit_range) ]
        yy = self._y[ (ep0-fit_range) : (ep0+fit_range) ]

        #   フィッティング
        try:
            initial = [self._x[ep0], self._y[ep0], fit_range]
            coef, pconv = curve_fit(gaussian, xx, yy, p0=initial)
        except RuntimeError:
            print("fail to fit on ", initial)
        else:
            #   フィッティング結果から頂点のx座標と包絡線を保存
            self.peak = (coef[1], gaussian(coef[1], *coef))
            self._y = [gaussian(i, coef[0], coef[1], coef[2]) for i in xx]
            self._x = xx

    def show(self, ax=None):
        if ax:
            ax.plot(self._x, self._y)
            ax.plot(self.peak[0], self.peak[1], 'o')
        print("ep: " + str(round(self.peak[0], 3)) + "um")
        return

class Fringes(object):
    def __init__(self, x=[], f=[], param=None):
        self.x = x
        self.f = f
        self.peaks = []
        self.param = param

    def find_peaks(self, threshold=0.5):
        #   包絡線極大値のインデックスのリストを求める
        env = abs(signal.hilbert(self.f))
        relmaxs = signal.argrelmax(env)[0]
        #   閾値を越えた極大値のみ処理
        self.peaks = []
        for relmax in relmaxs:
            if threshold < env[relmax]:
                ep = Envelope(self.x, env, relmax)
                self.peaks.append(ep)

    def down_sample(self, step):
        self.x = self.x[::step]
        self.f = self.f[::step]

    def show(self, ax = None):
        if ax:
            ax.plot(self.x, self.f)
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

    def __init__(self, fringe, fs):
        self.fringe = fringe
        self.fs = fs
        #   干渉縞を二乗
        self.fringe_sq_ = self.fringe * self.fringe


    def opt_calc(self, wave, point, cof_HeNe, wave_len):
        """
        HeNe干渉縞から位置を計算
        
        Parameters
            wave : array
                He-Ne干渉縞データ
            point : int
                求めたいポイントのインデックス
            cof_HeNe : float
                He-Ne干渉縞のスムージング用カットオフ周波数
            wave_len : float
                He-Neレーザの波長
        
        Returns
            position : float
                He-Ne干渉縞から計算したpointの相対位置
                光路長ベースでの計算
                
        """
        def search_n(point, points, position='n'):
            """
            ある点から最も近い点群中の点のインデックスを返す

            Parameters
                point : float
                    ある点
                points : float
                    点群
                position : string
                    左右の指定

            Returns
                point_opt : int
                    ある点から最も近い点の点群中でのインデックス

            Caution
            点群中の点に一致するものがあるときはその点のインデックスを返す
            """

            point_opt = np.argmin(np.abs(np.array(points) - point))
            # 最も近い点
            if position == 'n':
                return point_opt
            # 最も近い右側の点
            elif position == 'r':
                if points[point_opt] >= point:
                    return point_opt
                else:
                    return point_opt + 1
            # 最も近い左側の点
            elif position == 'l':
                if points[point_opt] <= point:
                    return point_opt
                else:
                    return point_opt - 1
            else:
                print('error')
                sys.exit()

        #   信号をスムージング
        self.laser_smooth_ = lpf(wave, self.fs, cof_HeNe)
        #   極大値を求める
        maxes = signal.argrelmax(self.laser_smooth_)[0]
        #   極小値を求める
        mins = signal.argrelmin(self.laser_smooth_)[0]
        #   内挿時のpointのy座標
        point_l = math.floor(point)
        f = interpolate.interp1d([point_l, point_l+1], [self.laser_smooth_[point_l], self.laser_smooth_[point_l+1]])
        y_point = f(point)

        #   point付近の極大、極小値を求める
        M0 = maxes[search_n(point, maxes, position='l')]
        M1 = maxes[search_n(point, maxes, position='r')]
        m0 = mins[search_n(point, mins, position='l')]
        m1 = mins[search_n(point, mins, position='r')]

        #   M1の位置を基準とした位相を求める
        # pointが極大値と一致するとき
        if M0 == M1:
            dn = 0
        # pointがM1に近いとき(M0と比べて)
        elif M0 <= m0 <= M1:
            mid = (self.laser_smooth_[M0] + self.laser_smooth_[m1]) / 2
            dn = -np.arccos((y_point - mid) / (self.laser_smooth_[M1] - mid))
        # pointがM0に近いとき(M1と比べて)
        else:
            mid = (self.laser_smooth_[M0] + self.laser_smooth_[m1]) / 2
            dn = -(np.pi * 2 - np.arccos((y_point - mid) / (self.laser_smooth_[M0] - mid)))
        # arccosの中身がちょうど-1になるときはpiを返す
        if np.isnan(dn):
            dn = -np.pi
        wave_number = list(maxes).index(M1) + dn / (2 * np.pi)
        return wave_number * wave_len
    @staticmethod
    def search_neighbourhood(target, points, position='n'):
        """ある点(target)から最も近い点群(points)中の点の番号を返す"""
        if position == 'n':  # 最も近い点
            list = []
            for i in range(len(points)):
                l = abs(points[i] - target)
                list.append(l)
            return np.argmin(list)
        elif position == 'r':  # 最も近い右側の点
            if points[np.searchsorted(points, target)] >= target:
                return np.searchsorted(points, target)
            else:
                return np.searchsorted(points, target) + 1
        elif position == 'l':  # 最も近い左側の点
            if points[np.searchsorted(points, target)] <= target:
                return np.searchsorted(points, target)
            else:
                return np.searchsorted(points, target) - 1
        else:
            print('error')
            sys.exit()

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
                fig_gauss = plt.figure(3)
                ax = fig_gauss.add_subplot(111)
                ax.plot(x, y, "ro", markersize=2)
                ax.plot(x, envelope, linewidth=3)
        else:
            print('定義されてない手法です')
            sys.exit()
        return eps


class Sest(object):
    """
    SESTアルゴリズムの実装

    Parameters
        X ; array
            干渉縞の標本データ(y座標)
        Y : array 
            干渉縞の標本データ(x座標)
        wl_c : float
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


def lpf(x, fs, fe):
    """
    ローパスフィルター

    Parameters
        x : array
            元の信号
        fs : float
            サンプリング周波数
        fe : float
            カットオフ周波数

    Returns
        y : array
            カットオフ後の信号
    """
    X = fftpack.fft(x)
    frecency = [abs(k * fs / len(x)) for k in range(len(x))]
    for k, f in enumerate(frecency):
        if (fe < f < frecency[-1] - fe) == 1:
            X[k] = complex(0, 0)
    y = np.real(fftpack.ifft(X))
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





def read_position(f_path):
    """
    ML-10のデータを読み込み

    Parameters
        f_path : string
            ML-10データのパス

    Returns
        position : float
            ML-10データの平均値
    """
    f = open(f_path)
    lines = f.readlines()
    data = []
    for line in lines:
        data.append(np.double(line))
    position = np.average(data)
    return position