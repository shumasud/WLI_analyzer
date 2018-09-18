from . import core as wli
import numpy as np
import pandas as pd
from scipy import interpolate
import sys
from scipy import signal
import math


class FringesScanner(object):
    def __init__(self):
        pass



def moving_average(y, num):
    b = np.ones(num) / num
    return np.convolve(y, b, mode='same')




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
    f = interpolate.interp1d([point_l, point_l + 1], [self.laser_smooth_[point_l], self.laser_smooth_[point_l + 1]])
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
        # sys.exit()

