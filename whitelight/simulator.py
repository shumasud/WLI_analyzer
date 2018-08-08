"""

"""

from pylab import *
from scipy import signal
from scipy.optimize import curve_fit
from . import base as wli


class Simulator(object):
    def __init__(self, *, lambda_c=1560 / 1000, band_w=25 / 1000, wl_step=1 / 1000, scan_w=100):
        self.lambda_c = lambda_c
        self.band_w = band_w
        self.wl_step = wl_step
        self.scan_w = scan_w

    def make_fringes(self, x, *, peaks=[0], pows=None, noise=[0, 0]):
        xn = x + noise[0]*np.random.randn(len(x))
        if not pows:
            pows = np.ones_like(peaks)
        f = np.zeros_like(xn)

        for (peak, pow) in zip(peaks, pows):
            xslice = np.where((peak - self.scan_w < xn) & (xn < peak + self.scan_w))
            f[xslice] += pow * self._make_fringe(xn[xslice] - peak)
        fn = f + noise[1]*np.random.randn(len(f))

        return wli.Fringes(x, fn)

    def _make_fringe(self, x, l_bs=0, offset=0, material='BK7'):
        fringe_list = []
        wls = self.lambda_c + np.arange(-self.band_w, self.band_w, self.wl_step)  # 波長のリスト
        print("making fringes")
        for wl in wls:  # 各波長で干渉縞作成
            k_i = 2 * np.pi / wl  # 波数k
            phi_r = self._phase_shift(wl, material)  # 反射での位相シフト
            opd_bs = (self._ref_index(wl, 'BK7') - self._ref_index(self.lambda_c, 'BK7')) * l_bs
            # 光路はダブルパス
            phi = phi_r + k_i * 2 * (offset + opd_bs + x)
            fringe = self._I_gauss(wl) * np.cos(phi)
            fringe_list.append(fringe)
            print(".", end="")
        print("done")
        fringes = np.array(fringe_list)
        fringe_total = np.sum(fringes, axis=0)  # それぞれの波長での干渉縞を重ね合わせ
        return fringe_total / max(fringe_total)

    def _I_gauss(self, wl):
        # scipy.signal.gausspulse
        sigma2 = (self.band_w ** 2) / (8 * np.log(2))
        I = np.exp(-((wl - self.lambda_c) ** 2) / (2 * sigma2)) / (np.power(2 * np.pi * sigma2, 0.5))
        return I

    def _phase_shift(self, wl, material='BK7'):
        params = {}
        params['Ag'] = (1.2104, -1.3392, 6.8276, 0.1761)
        params['Fe'] = (0.5294, -2.7947, 2.7647, 1.3724)
        params['Al'] = (1.3394, -0.6279, 11.297, -1.5539)
        params['Au'] = (0.6118, -0.3893, 6.4455, -0.1919)

        if material == 'BK7':
            phi = np.pi
        else:
            param = params[material]
            n = param[0] * wl + param[1]
            k = param[2] * wl + param[3]
            phi = np.arctan(-2 * k / (n * n + k * k - 1))
        return phi

    def _ref_index(self, wl, material='BK7'):
        if material == 'BK7':
            B1 = 1.03961212E+00
            B2 = 2.31792344E-01
            B3 = 1.01046945E+00
            C1 = 6.00069867E-03
            C2 = 2.00179144E-02
            C3 = 1.03560653E+02
            wl2 = wl ** 2
            n = np.sqrt(1 + B1 * wl2 / (wl2 - C1) + B2 * wl2 / (wl2 - C2) + B3 * wl2 / (wl2 - C3))
        else:
            n = -8e-10 * wl + 1.0003

        return n
