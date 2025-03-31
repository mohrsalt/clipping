import numpy as np
import tqdm

from utils import CheckOnOffFixedDeg, check_on_off


def main(out_height, out_width):
    array = np.zeros((360, 180, 3))
    deg = 0
    check_on_off_fixed_deg = CheckOnOffFixedDeg(deg, out_height, out_width)

    for azi in tqdm.tqdm(range(0, 360, 1)):
        for ele in range(0, 180, 1):
            array[azi, ele] = check_on_off_fixed_deg.check(azi, ele)
            # array[azi, ele] = check_on_off(deg, azi, ele, out_height, out_width)  # same as above but slow, keep as reference

    out_file = "./aziele2valuexy_0deg_ufov100w{}h{}.npy".format(out_width, out_height)
    np.save(out_file, array)


if __name__ == "__main__":
    out_height, out_width = 360, 640

    main(out_height, out_width)
