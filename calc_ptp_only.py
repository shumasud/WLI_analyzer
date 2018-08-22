# -*- coding: utf-8 -*-
"""
Name:
    calc_ptp.py
Purpose:
    白色干渉縞のピーク間距離を求める
Specification:
    スクリプト
Environment:
    Python 3.6.0

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import whitelight.core as wl
plt.style.use('seaborn')


def write_list(result, c_name, file='data\\result.csv'):
    """結果をCSVファイルに書き込み"""
    df = pd.DataFrame(result)
    df.columns = c_name
    df.to_csv(file)


def read_fringe(f_path, start=0, end=0, skip_rows=0, *, mode='csv'):
    """
    干渉縞データの読み込み

    Parameters
        f_path : string
            干渉縞データファイルのパス

    Returns
        data1 : array
            一列目のデータ
        data2 : array
            二列目のデータ
    """
    data = []
    if mode == 'csv':
        df = pd.read_csv(f_path, names=['data1', 'data2'])
        data.append(df.iloc[start + skip_rows:end, 0].as_matrix().astype(np.float))
        data.append(df.iloc[start + skip_rows:end, 1].as_matrix().astype(np.float))
    elif mode == 'bin':
        if end <= 0:
            length = -1
        else:
            length = end
        f = open(f_path, 'rb')
        ds = np.fromfile(f, np.dtype('<i4'), 2)
        db = np.fromfile(f, np.dtype('<d'), length)
        f.close()
        data = np.reshape(db, ds).tolist()

    return data


def moving_average(y, num):
    b = np.ones(num) / num
    return np.convolve(y, b, mode='same')


if __name__ == '__main__':
    """
    Parameters
        start : int
            開始点
        end : int
            終了点
        fs : float
            サンプリング周波数
            ポイントベースの計算では1
        f_path_list : string
            計算したい干渉縞データパスのリスト
        cof_env : float
            包絡線を求める際のカットオフ周波数
        ep_sens : float
            包絡線の極大値への感度
            この値以上の極大値に反応
        method : string
            包絡線ピークを求める方法
            'SL'は二乗＋ローパスフィルタ
            'HG'はヒルベルト変換＋ガウシアンフィッティング
        f_rate : float
            HG法でフィッティングする際のフィッティング率
            包絡線ピークの近傍でピークのパワーのf_rate倍以上の範囲を用いてフィッティング

    """
    start = 0
    end = 0
    fs = 100E3
    f_path_head = 'data\\b6\\'
    f_path_list = [str(i) for i in range(162700, 162708)]
    f_path_foot = '.bin'
    cof_env = 30
    ep_sens = 0.3
    method = 'HG'
    f_rate = 0.5
    calculation = True
    plot = True

    result = []
    for f_path in f_path_list:
        try:
            """データの読み込み"""
            fringe, sig_DS = read_fringe(f_path_head + f_path + f_path_foot, start, end, skip_rows=0, mode='bin')

            """データ取得失敗回避"""
            if fringe[0] == fringe[1] == 0:
                raise SystemError('Error message')

            """干渉縞データをクラスに適合"""
            ave_fringe = np.average(fringe)
            fringe = fringe - ave_fringe
            wave = wl.WhiteLight(fringe, fs)
            print(f_path)

            # """変位信号を低次関数で近似"""
            # x_DS = np.arange(0, len(sig_DS), 1)
            # coef_reg = np.polyfit(x_DS, sig_DS, 1)
            # print(coef_reg[0])

            """変位信号を移動平均"""
            sig_DS_MA = moving_average(sig_DS, 300)

            if plot:
                """時間(空間）ドメインのグラフをプロット"""
                fig_time = plt.figure(1)
                ax11 = fig_time.add_subplot(111)
                #   白色干渉縞をプロット
                ax11.set_title('White Light Fringe')
                ax11.plot(wave.fringe)
                ax11.plot(sig_DS)
                ax11.plot(sig_DS_MA)

            if calculation:
                # 包絡線ピークのインデックスを求める
                eps = wave.calc_EPs(cof_env, ep_sens, method=method, f_rate=f_rate)
                dis_eps = [sig_DS_MA[int(eps[i+1])] - sig_DS_MA[int(eps[i])] for i in range(0, len(eps)-1)]
                dis_ep = [dis_ep for dis_ep in dis_eps if dis_ep > 0.1][0]
                time = int(f_path[0:2])*60*60 + int(f_path[2:4])*60 + int(f_path[4:6])
                result.append([time, dis_ep])
                if plot:
                    ax11.plot(wave.envelope_, linewidth=2)
                    ax11.plot(int(eps[0]), wave.envelope_[int(eps[0])], "ro", markersize=10)
                    ax11.plot(int(eps[1]), wave.envelope_[int(eps[1])], "ro", markersize=10)
            if plot:
                """スペクトルをプロット"""
                fig_freq = plt.figure(2)
                #   白色干渉縞をプロット
                ax21 = fig_freq.add_subplot(211)
                wave.spe_ana(wave.fringe, ax21)
                ax21.set_title('Spectrum of White Light Fringe')
                #   白色干渉縞の二乗信号をプロット
                ax22 = fig_freq.add_subplot(212)
                wave.spe_ana(wave.fringe_sq_, ax22)
                ax22.set_title('Spectrum of Squared White Light Fringe')
        except Exception as err:
            print(err)
            pass
    # write_list(result, ['time', 'ptp'])
    if plot:
        plt.show()




