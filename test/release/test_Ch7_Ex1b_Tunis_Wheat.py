#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import test_utils
from .test_utils import read_yield


def test_Ch7_Ex1b_Tunis_Wheat(tmpdir, context):
    # aqpy_yld, aos_yld = test_utils.read_yield(tmpdir)
    aqpy_yld, aos_yld = read_yield(tmpdir)
    np.testing.assert_almost_equal(aqpy_yld, aos_yld, 2)
