from pylab import *
from scipy import signal
from scipy.optimize import curve_fit
from . import base as wl


class Light(wl.Fringes):
    """
    パラメータ
    ------------
    wl_c : (val) 中心波長[um]
    wl_bw : (val) 波長のバンド幅[um]
    wl_step : (val) 波長のステップ幅[um]

    属性
    ------------
    scale_ : (array) 走査鏡の変位[um]
    fringe_ : (array) 干渉縞[um]
    envelope_ : (array) 包絡線[um]

    """

    def __init__(self, wl_c=1560/1000, wl_bw=25/1000, wl_step=1/1000):
        super().__init__()
        self.wl_c = wl_c
        self.wl_bw = wl_bw
        self.wl_step = wl_step
        self.wl_list_ = np.arange(wl_c - wl_bw / 2 * 2, (wl_c + wl_step) + wl_bw / 2 * 2, wl_step)  # 波長のリスト


    @staticmethod
    def ref_index_air(wl):
        n = -8e-10 * wl + 1.0003
        return n

    @staticmethod
    def ref_index_BK7(wl):
        B1 = 1.03961212E+00
        B2 = 2.31792344E-01
        B3 = 1.01046945E+00
        C1 = 6.00069867E-03
        C2 = 2.00179144E-02
        C3 = 1.03560653E+02
        wl2 = wl**2

        n = np.sqrt( 1 + B1*wl2 / (wl2 - C1) + B2*wl2 / (wl2 - C2) + B3*wl2 / (wl2 - C3) )
        return n

    @staticmethod
    def phase_shift(wl, material):
        params = {}
        params['Ag'] = (1.2104, -1.3392, 6.8276, 0.1761)
        params['Fe'] = (0.5294, -2.7947, 2.7647, 1.3724)
        params['Al'] = (1.3394, -0.6279, 11.297, -1.5539)
        params['Au'] = (0.6118, -0.3893, 6.4455, -0.1919)

        param = params[material]
        n = param[0] * wl + param[1]
        k = param[2] * wl + param[3]
        phi = np.arctan(-2 * k / (n * n + k * k - 1))
        return phi

    def I_gauss(self, wl):
        sigama2 = (self.wl_bw ** 2) / (8 * np.log(2))
        f = np.exp(-((wl - self.wl_c) ** 2) / (2 * sigama2)) / (np.power(2 * np.pi * sigama2, 0.5))
        return f

    def make_scale(self, scan_len, scan_step):
        self.x = np.arange(-scan_len / 2, scan_len / 2 + scan_step, scan_step)

    def make_scale_noised(self, jitter, grad):
        self.x = jitter * randn(len(self.x)) + (1 + grad) * self.x

    def make_fringe_noised(self, noise, drift):
        a0 = noise * randn(len(self.x))
        a1 = drift / max(self.x) * self.x
        self.f = self.f + a0 + a1

    def make_fringe(self, l_ref=3000 * 1000, l_bs=0, offset=0, material='BK7'):
        """スケールと干渉縞を作成"""
        fringe_list = []
        for wl in self.wl_list_:
            """あるwlでの干渉縞を作成"""
            print("making fringe")

            k_i = 2 * np.pi / wl
            intensity = self.I_gauss(wl)

            phi_x = k_i * self.x * 2
            if material == 'BK7':
                phi_r = np.pi
            else:
                phi_r = self.phase_shift(wl, material)  # 反射での位相シフト(ガラス以外)
            phi_bs = k_i * (self.ref_index_BK7(wl) - self.ref_index_BK7(self.wl_c)) * l_bs * 2
            phi_offset = k_i * offset * 2
            phi = list(map(lambda x: x - phi_r - phi_bs - phi_offset + np.pi, phi_x))
            fringe = intensity * np.cos(phi)
            fringe_list.append(fringe)

        print("done")
        fringes = np.array(fringe_list)
        fringe_total = np.sum(fringes, axis=0)  # それぞれの波長での干渉縞を重ね合わせ
        self.f = fringe_total / max(fringe_total)

