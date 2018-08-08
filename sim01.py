import copy
import matplotlib.pyplot as plt
import whitelight.simulator
import numpy as np

if __name__ == '__main__':
    sim_param = {'lambda_c': 1560 / 1000,
                 'band_w': 25 / 1000,
                 'wl_step': 0.1 / 1000,
                 'scan_w': 200}
    scan_step = 0.5 / 1000

    # initialize WLI fringes simulator
    ws = whitelight.simulator.Simulator(**sim_param)
    x = np.arange(0, 500, scan_step)

    # generate fringes
    wav1 = ws.make_fringes(x, [100, 400])
    wav2 = copy.deepcopy(wav1)
    wav2.down_sample(100)

    # plot
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)
    wav1.find_peaks()
    wav2.find_peaks()
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
