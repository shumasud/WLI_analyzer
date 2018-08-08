import copy
import matplotlib.pyplot as plt
import whitelight.simulator
import numpy as np

plt.style.use('seaborn')

if __name__ == '__main__':
    SIM_PARAM = {'lambda_c': 1560 / 1000,
                 'band_w': 25 / 1000,
                 'wl_step': 0.1 / 1000,
                 'scan_w': 300,
                 'scan_step': 10 / 1000}

    noise = [1/1000, 5/1000]  # noise_x, noise_f
    noise = [0, 0]
    peaks = [250, 350]

    # initialize WLI fringes simulator
    ws = whitelight.simulator.Simulator(**SIM_PARAM)

    # generate fringes
    wav1 = ws.make_fringes(peaks=peaks, noise=noise)
    wav2 = copy.deepcopy(wav1)
    wav2.down_sample(10)

    # plot
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    wav1.find_peaks(0.2)
    wav2.find_peaks(0.2)
    wav1.show(ax1)
    wav2.show(ax2)
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
