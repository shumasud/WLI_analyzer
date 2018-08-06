# -*- coding: utf-8 -*-
"""
Name:
    simulator.py
Purpose:
    白色干渉のシミュレーター
Specification:
    
Environment:
    Python 3.5.1
    
"""
import copy
import matplotlib.pyplot as plt
import whitelight.simulator as ws

if __name__ == '__main__':



    wl_c = 1560 / 1000  # 中心波長[um]
    wl_bw = 25 / 1000  # バンド幅(FWHM)[um]
    scan_len = 100  # スキャン長さ[um]
    scan_step = 1 / 1000
    l_bs = 0  # BSの長さ[um]
    offset = 0

    # 基準干渉縞作成
    light = ws.Light(wl_c, wl_bw, wl_step=10 / 1000)
    light.make_scale(scan_len, scan_step)
    light.make_scale_noised(0.000/1000, 0/1000)
    light.make_fringe(l_bs=l_bs, offset=offset, material='BK7')
    light.make_fringe_noised(0.000/1000, 0)

    # 干渉縞複製・計算
    light2 = copy.deepcopy(light)
    light2.down_sample(100)
    light3 = copy.deepcopy(light)
    light3.make_scale_noised(100/1000, 0/1000)
    # light3.make_fringe(l_bs=l_bs, offset=offset, material='BK7')
    light3.down_sample(100)

    light.find_peaks()
    light2.find_peaks()
    light3.find_peaks()

    # 表示
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    light.show(ax1)
    light2.show(ax2)
    light3.show(ax3)

    plt.show()


    # def write_list(result, c_name, file='result.csv'):
    #     """結果をCSVファイルに書き込み"""
    #     df = pd.DataFrame(result)
    #     df.columns = c_name
    #     df.to_csv(file)
    #    write_list([[x, y] for x, y in zip(light.scale_, light.fringe_)], ['position', 'intensity'])



    """L_bsを変更しながら計算"""
    """
    peaks = []
    for i in range(100):
        l_bs = i * 10
        light.make_fringe(l_bs=l_bs)
        light.peak_detect()
        peak = [round(light.scale_[light.ep_], 3), round(light.scale_[light.fp_], 3)]
        peaks.append(peak)
    write_list(peaks)
    """
